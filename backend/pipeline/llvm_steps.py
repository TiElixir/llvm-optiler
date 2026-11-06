"""LLVM and below: opt (O3) -> llc -> link -> objdump.

The textual LLVM IR comes from MLIR (`translate_module_to_llvmir`); everything after
that is plain LLVM tooling, so the `asm` stage is the disassembly of the exact
``bin/model.so`` that gets benchmarked.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.toolchain import find_tool, require_tool, target_triple

from .mlir_runner import SnapshotRecord

RT_SRC = Path(__file__).resolve().parent.parent / "runtime" / "optiler_rt.c"
OPT_PIPELINE = "default<O3>"
# Per-pass dumps re-print every embedded weight; past this many bytes of global initializers
# the dump run is skipped (input and O3 result are still stored).
LARGE_GLOBALS_BYTES = 2_000_000

_DUMP_RE = re.compile(r"^\*\*\* IR Dump (After|Before) (.+?) \*\*\*\s*$")


class BuildError(RuntimeError):
    def __init__(self, step: str, cmd: list[str], output: str):
        super().__init__(f"{step} failed: {' '.join(cmd[:3])} ...\n{output[-2000:]}")
        self.step, self.cmd, self.output = step, cmd, output


def _run(step: str, cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if r.returncode != 0:
        raise BuildError(step, cmd, (r.stderr or r.stdout))
    return r


@dataclass
class BuildResult:
    records: list[SnapshotRecord] = field(default_factory=list)
    so_path: str = "bin/model.so"
    asm_path: str = "asm/model.s"
    commands: list[list[str]] = field(default_factory=list)
    times_ms: dict[str, float] = field(default_factory=dict)
    per_pass_dumps: bool = True


def parse_print_changed(text: str) -> list[tuple[str, str, str]]:
    """Split ``-print-after-all``-style stderr into ``(kind, header, ir_text)`` blocks."""
    blocks: list[tuple[str, str, str]] = []
    cur: list[str] | None = None
    head = ("", "")
    for line in text.splitlines():
        m = _DUMP_RE.match(line)
        if m:
            if cur is not None:
                blocks.append((head[0], head[1], "\n".join(cur) + "\n"))
            head, cur = (m.group(1), m.group(2)), []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append((head[0], head[1], "\n".join(cur) + "\n"))
    return blocks


def elide_globals(text: str, limit: int = 240) -> str:
    """Shorten huge global initializers (weights) so stored snapshots stay readable."""
    out = []
    for line in text.split("\n"):
        if line.startswith("@") and len(line) > limit:
            out.append(f"{line[:limit]} ... <elided {len(line) - limit} chars>")
        else:
            out.append(line)
    return "\n".join(out)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:40]


def build_native(run_dir: Path, llvm_ir: str, start_idx: int, cpu: str = "native",
                 on_step=None) -> BuildResult:
    run_dir = Path(run_dir)
    opt, llc = require_tool("opt"), require_tool("llc")
    clang, objdump = require_tool("clang"), require_tool("llvm-objdump")
    res = BuildResult()
    for d in ("llvm/full", "bin", "asm", "reports"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    def step(name: str) -> None:
        if on_step:
            on_step(name)

    # 1. input IR
    ll_in = run_dir / "llvm" / "full" / "input.ll"  # real IR for opt
    ll_in.write_text(llvm_ir)
    sha = hashlib.sha1(llvm_ir.encode()).hexdigest()
    disp = run_dir / "llvm" / f"{start_idx:03d}_input.ll"  # elided copy for display
    disp.write_text(elide_globals(llvm_ir))
    res.records.append(SnapshotRecord(
        idx=start_idx, stage="llvm_ir", pass_name="translate-to-llvmir", pass_text="translate-to-llvmir",
        composite=False, changed=True, time_ms=0.0, op_counts={}, total_ops=llvm_ir.count("\n"),
        path=str(disp.relative_to(run_dir)), sha1=sha, kind="import"))

    # 2. opt -O3, per-pass changed dumps
    step("opt")
    opt_out = run_dir / "llvm" / "full" / "model_opt.ll"
    remarks = run_dir / "reports" / "remarks.yaml"
    globals_bytes = sum(len(l) for l in llvm_ir.split("\n") if l.startswith("@"))
    per_pass = globals_bytes <= LARGE_GLOBALS_BYTES
    res.per_pass_dumps = per_pass
    cmd = [opt, "-S", f"-passes={OPT_PIPELINE}", f"-mcpu={cpu}"]
    if per_pass:
        cmd += ["-print-changed=quiet"]  # function passes dump the function, module passes the module
    cmd += ["-pass-remarks-missed=loop-vectorize", "-pass-remarks-analysis=loop-vectorize",
            f"-pass-remarks-output={remarks}", str(ll_in), "-o", str(opt_out)]
    t0 = time.perf_counter()
    r = _run("opt", cmd)
    res.times_ms["opt"] = (time.perf_counter() - t0) * 1000
    res.commands.append(cmd)
    (run_dir / "reports" / "opt_stderr.txt").write_text(r.stderr)
    (run_dir / "reports" / "remarks.txt").write_text(
        "\n".join(l for l in r.stderr.splitlines() if "remark:" in l) + "\n")
    idx = start_idx + 1
    prev = sha
    for kind, header, text in (parse_print_changed(r.stderr) if per_pass else []):
        if kind != "After":
            continue
        name = header.split(" on ")[0].strip()
        p = run_dir / "llvm" / f"{idx:03d}_{_slug(name)}.ll"
        p.write_text(elide_globals(text))
        s = hashlib.sha1(text.encode()).hexdigest()
        res.records.append(SnapshotRecord(
            idx=idx, stage="llvm_ir", pass_name=name, pass_text=header, composite=False,
            changed=s != prev, time_ms=0.0, op_counts={}, total_ops=text.count("\n"),
            path=str(p.relative_to(run_dir)), sha1=s, kind="pass"))
        prev, idx = s, idx + 1

    final = opt_out.read_text()
    fin = run_dir / "llvm" / f"{idx:03d}_O3-result.ll"
    fin.write_text(elide_globals(final))
    fs = hashlib.sha1(final.encode()).hexdigest()
    res.records.append(SnapshotRecord(
        idx=idx, stage="llvm_ir", pass_name="O3 result", pass_text="opt -passes=default<O3>",
        composite=False, changed=fs != prev, time_ms=res.times_ms["opt"], op_counts={},
        total_ops=final.count("\n"), path=str(fin.relative_to(run_dir)), sha1=fs, kind="pass"))
    idx += 1

    # 3. llc -> object
    step("llc")
    obj = run_dir / "bin" / "model.o"
    cmd = [llc, "-O3", f"-mcpu={cpu}", "-relocation-model=pic", "-filetype=obj",
           str(opt_out), "-o", str(obj)]
    t0 = time.perf_counter()
    _run("llc", cmd)
    res.times_ms["llc"] = (time.perf_counter() - t0) * 1000
    res.commands.append(cmd)

    # 4. runtime + link
    step("link")
    rt_obj = run_dir / "bin" / "optiler_rt.o"
    cmd = [clang, "-O2", "-fPIC", "-c", str(RT_SRC), "-o", str(rt_obj)]
    _run("clang(rt)", cmd)
    res.commands.append(cmd)
    so = run_dir / "bin" / "model.so"
    cmd = [clang, "-shared", "-o", str(so), str(obj), str(rt_obj), "-lm"]
    _run("link", cmd)
    res.commands.append(cmd)

    # 5. disassembly of the exact .so
    step("asm")
    cmd = [objdump, "-d", "-l", "--no-show-raw-insn", "-C", str(so)]
    r = _run("llvm-objdump", cmd)
    asm = run_dir / "asm" / "model.s"
    asm.write_text(r.stdout)
    res.commands.append(cmd)
    s = hashlib.sha1(r.stdout.encode()).hexdigest()
    res.records.append(SnapshotRecord(
        idx=idx, stage="asm", pass_name="llvm-objdump", pass_text=" ".join(cmd), composite=False,
        changed=True, time_ms=0.0, op_counts={}, total_ops=r.stdout.count("\n"),
        path=str(asm.relative_to(run_dir)), sha1=s, kind="import"))
    return res


def llvm_tool_versions() -> dict:
    out = {}
    for t in ("clang", "opt", "llc", "llvm-objdump"):
        p = find_tool(t)
        if p:
            m = re.search(r"version ([0-9][^\s]*)", subprocess.run([p, "--version"], capture_output=True, text=True).stdout)
            out[t] = m.group(1) if m else "?"
    out["triple"] = target_triple()
    return out
