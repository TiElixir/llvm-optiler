"""Detector framework. A detector that is not certain emits nothing (PLAN.md section 9)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.provenance.mlir_ops import walk
from backend.schemas import FxNode, Impact, Metrics, StageOps

MIN_RATIO = 1.05  # findings below this estimated speedup are not worth showing


@dataclass
class Context:
    run_dir: Path
    stages: dict[str, StageOps]
    metrics: Metrics
    fx_nodes: list[FxNode]
    pipeline_yaml: str = ""
    linalg: list[dict] = field(default_factory=list)   # live linalg info at the end of linalg_tensor
    copies: list[dict] = field(default_factory=list)   # memref.copy ops at the end of linalg_buffer
    calib: dict = field(default_factory=dict)
    lanes: int = 8

    @property
    def runtime_ms(self) -> float:
        return self.metrics.runtime_ms_median or 0.0

    def code(self, line: int) -> str | None:
        n = next((n for n in self.fx_nodes if n.line == line and n.code), None)
        return n.code if n else None

    def stage_ops(self, stage: str) -> list:
        s = self.stages.get(stage)
        return s.ops if s else []

    def share(self, line: int) -> float:
        return (self.metrics.per_line_share or {}).get(str(line), 0.0)


def est(ratio: float, note: str | None = None) -> Impact:
    return Impact(ratio=round(ratio, 3), kind="est", note=note)


def amdahl(saved_fraction: float) -> float:
    """Speedup if ``saved_fraction`` of the runtime disappears."""
    return 1.0 / max(1.0 - min(saved_fraction, 0.95), 0.05)


# ---------------------------------------------------------------- remarks

_ENTRY = re.compile(r"^--- !(\w+)\n(.*?)(?=^--- |\Z)", re.S | re.M)


def parse_remarks(path: Path) -> list[dict]:
    """opt -pass-remarks-output (YAML with !Missed/!Passed tags) -> [{kind, pass, name, line, message}]."""
    if not path.exists():
        return []
    out = []
    for kind, body in _ENTRY.findall(path.read_text()):
        g = lambda k: (re.search(rf"^{k}:\s+(.*)$", body, re.M) or [None, ""])[1].strip()  # noqa: E731
        line = re.search(r"Line:\s*(\d+)", body)
        msg = " ".join(re.findall(r"String:\s*'?(.*?)'?\s*$", body, re.M))
        out.append({"kind": kind, "pass": g("Pass"), "name": g("Name"), "line": int(line.group(1)) if line else None,
                    "message": msg})
    return out


# ---------------------------------------------------------------- live linalg info (needs the module)

def collect_linalg_info(module) -> list[dict]:
    """For each linalg op at the end of linalg_tensor: walk index, name, iterator types, result type, users."""
    ops = walk(module.operation)
    index_of = {hash(o): i for i, (o, _d, _p, _pl) in enumerate(ops)}
    info = []
    for i, (op, _d, _p, payload) in enumerate(ops):
        if payload or not op.name.startswith("linalg."):
            continue
        its = []
        if "iterator_types" in op.attributes:
            its = re.findall(r"parallel|reduction|window", str(op.attributes["iterator_types"]))
        users = []
        for r in op.results:
            for u in r.uses:
                owner = getattr(u.owner, "operation", None)
                if owner is not None and hash(owner) in index_of:
                    users.append(index_of[hash(owner)])
        info.append({"idx": i, "name": op.name, "iterators": its, "result": str(op.results[0].type) if op.results else None,
                     "users": users})
    return info


_SHAPE = re.compile(r"(?:tensor|memref)<((?:\d+x)+)f32")


def type_bytes(t: str | None) -> int | None:
    """Bytes of a static f32 tensor/memref type string, or None."""
    m = _SHAPE.search(t or "")
    if not m:
        return None
    n = 1
    for d in m.group(1).rstrip("x").split("x"):
        n *= int(d)
    return n * 4


def collect_copy_info(module) -> list[dict]:
    """memref.copy ops at the end of linalg_buffer: size, and whether the source is a temporary (an alloc)."""
    ops = walk(module.operation)
    out = []
    for i, (op, _d, _p, _pl) in enumerate(ops):
        if op.name != "memref.copy":
            continue
        src = op.operands[0]
        owner = getattr(src.owner, "operation", None)
        out.append({"idx": i, "bytes": type_bytes(str(src.type)), "src_is_alloc": owner is not None and owner.name == "memref.alloc"})
    return out


def memory_ms(nbytes: float, calib: dict) -> float:
    """Time to read + write ``nbytes`` once at the measured bandwidth (cache bandwidth when it fits)."""
    bw = calib.get("bandwidth_cache_gbs" if nbytes <= 262144 else "bandwidth_gbs") or 20.0
    return 2 * nbytes / (bw * 1e9) * 1e3
