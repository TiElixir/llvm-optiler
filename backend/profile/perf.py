"""perf-based runtime attribution (PLAN.md 8.2).

``perf record`` samples the compiled model in a loop; each sample's first frame inside ``model.so``
(the sampled instruction, or the caller of libm's ``expf`` and friends) is resolved through the DWARF line
table to a line of ``anchor/llvm_dialect.mlir``, i.e. to an LLVM-dialect op, i.e. to source lines.
"""
from __future__ import annotations

import bisect
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from backend.toolchain import find_tool, perf_status, require_tool

ROOT = Path(__file__).resolve().parents[2]
FREQ = 4000
_FRAME = re.compile(r"^\s*([0-9a-f]+)\s+(.+?)\s+\((.*)\)\s*$")
_ANCHOR = re.compile(r"llvm_dialect\.mlir:(\d+)")


class PerfUnavailable(RuntimeError):
    pass


@dataclass
class Attribution:
    mode: str                                   # perf call-graph mode used: lbr | dwarf | none
    total: int = 0                              # samples with a frame in model.so
    other_samples: int = 0                      # samples in model.so with no anchor line
    by_anchor_line: Counter = field(default_factory=Counter)
    by_addr: Counter = field(default_factory=Counter)   # module-relative instruction address -> samples
    outside: int = 0                            # samples with no model.so frame (interpreter startup, ...)

    @property
    def other_share(self) -> float:
        return self.other_samples / self.total if self.total else 0.0


def record(run_dir: Path, seconds: float = 1.5, core: int | None = None) -> tuple[Path, str]:
    """Record the compiled model. Returns (perf.data path, call-graph mode)."""
    st = perf_status()
    if not st["available"]:
        raise PerfUnavailable(st.get("reason") or "perf is not available")
    from backend.pipeline.run import bench_core  # local import: avoids a cycle at module load

    out = run_dir / "perf"
    out.mkdir(exist_ok=True)
    data = out / "perf.data"
    cmd_tail = [sys.executable, "-P", "-m", "backend.execute.perf_loop", str(run_dir), "--seconds", str(seconds)]
    pin = [find_tool("taskset"), "-c", str(core if core is not None else bench_core())] if find_tool("taskset") else []
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    last_err = ""
    for mode in ("lbr", "dwarf,8192", None):
        cg = ["--call-graph", mode] if mode else []
        cmd = [st["path"], "record", "-q", "-e", "cycles:u", "-F", str(FREQ), *cg, "-o", str(data), "--", *pin, *cmd_tail]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        if r.returncode == 0 and data.exists() and data.stat().st_size > 0:
            return data, (mode.split(",")[0] if mode else "none")
        last_err = (r.stderr or r.stdout).strip().splitlines()[-1:] and (r.stderr or r.stdout).strip().splitlines()[-1] or ""
    raise PerfUnavailable(f"perf record failed: {last_err}")


def symbol_table(so: Path) -> dict[str, int]:
    """name -> address of the defined text symbols of ``so``."""
    r = subprocess.run([require_tool("llvm-nm"), "--defined-only", "-n", str(so)], capture_output=True, text=True)
    table: dict[str, int] = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "tTwW":
            table[parts[2]] = int(parts[0], 16)
    return table


def parse_script(text: str, so_name: str, symtab: dict[str, int]) -> tuple[list[int], int]:
    """One module-relative address per sample that has a frame in the model (return addresses minus 1)."""
    addrs: list[int] = []
    outside = 0
    for block in re.split(r"\n\s*\n", text):
        frames = [m.groups() for line in block.splitlines() if (m := _FRAME.match(line))]
        hit = None
        for depth, (_ip, sym, dso) in enumerate(frames):
            if dso.endswith(so_name):
                name, _, off = sym.rpartition("+0x")
                if not name or name not in symtab:
                    continue
                hit = symtab[name] + int(off, 16) - (1 if depth > 0 else 0)  # caller frames hold return addresses
                break
        if hit is None:
            if frames:
                outside += 1
        else:
            addrs.append(hit)
    return addrs, outside


def symbolize(so: Path, addrs: list[int]) -> dict[int, int | None]:
    """module address -> line in anchor/llvm_dialect.mlir (None when the line table has no entry)."""
    uniq = sorted(set(addrs))
    if not uniq:
        return {}
    r = subprocess.run([require_tool("llvm-symbolizer"), f"--obj={so}", "--inlines"],
                       input="\n".join(f"0x{a:x}" for a in uniq) + "\n", capture_output=True, text=True, timeout=120)
    blocks = r.stdout.split("\n\n")
    out: dict[int, int | None] = {}
    for a, blk in zip(uniq, blocks):
        m = _ANCHOR.search(blk)
        out[a] = int(m.group(1)) if m else None
    return out


def attribute(run_dir: Path, seconds: float = 1.5, so: str = "bin/model.so") -> Attribution:
    data, mode = record(run_dir, seconds)
    perf = find_tool("perf")
    r = subprocess.run([perf, "script", "-i", str(data), "-F", "ip,sym,symoff,dso"],
                       capture_output=True, text=True, timeout=300)
    so_path = run_dir / so
    symtab = symbol_table(so_path)
    addrs, outside = parse_script(r.stdout, "model.so", symtab)
    lines = symbolize(so_path, addrs)
    att = Attribution(mode=mode, total=len(addrs), outside=outside)
    for a in addrs:
        att.by_addr[a] += 1
        ln = lines.get(a)
        if ln is None:
            att.other_samples += 1
        else:
            att.by_anchor_line[ln] += 1
    return att


def nearest_instruction(sorted_addrs: list[int], addr: int) -> int | None:
    """Index of the instruction at or just before ``addr`` (a return address minus 1 lands inside a call)."""
    i = bisect.bisect_right(sorted_addrs, addr) - 1
    return i if i >= 0 else None
