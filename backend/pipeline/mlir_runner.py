"""Run an MLIR pipeline one pass at a time, snapshotting the module after each pass."""
from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from torch_mlir import ir
from torch_mlir.dialects import llvm as llvm_d
from torch_mlir.passmanager import PassManager

from .config import PipelineConfig, expand, pass_name


@dataclass
class SnapshotRecord:
    idx: int
    stage: str
    pass_name: str
    pass_text: str
    composite: bool
    changed: bool
    time_ms: float
    op_counts: dict[str, int]
    total_ops: int
    path: str
    sha1: str
    kind: str = "pass"  # import | pass | instrumentation

    def to_dict(self) -> dict:
        return asdict(self)


class PipelineError(RuntimeError):
    def __init__(self, stage: str, pass_text: str, message: str):
        super().__init__(f"{stage}: {pass_text}: {message}")
        self.stage, self.pass_text, self.message = stage, pass_text, message


def op_counts(module: ir.Module) -> Counter:
    counts: Counter = Counter()

    def walk(op) -> None:
        counts[op.name] += 1
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner.operation)

    for region in module.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                walk(inner.operation)
    return counts


ELIDE_LIMIT = 16  # constants with more elements are elided in stored snapshots


def print_module(module: ir.Module, elide: bool = True) -> str:
    """Text of the module with debug info. Elided text is for display only (not re-parseable)."""
    if not elide:
        return module.operation.get_asm(enable_debug_info=True)
    return module.operation.get_asm(
        enable_debug_info=True, large_elements_limit=ELIDE_LIMIT, large_resource_limit=ELIDE_LIMIT)


def full_sha(module: ir.Module) -> str:
    """Hash of the complete IR (constants included) so constant folding counts as a change."""
    return hashlib.sha1(print_module(module, elide=False).encode()).hexdigest()


def parse_and_print(ctx: ir.Context) -> Callable[[str], str]:
    def f(spec: str) -> str:
        with ctx:
            return str(PassManager.parse(f"builtin.module({spec})"))
    return f


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:48]


def snapshot(module, out_dir: Path, idx: int, stage: str, name: str, text: str,
             prev_sha: str | None, **kw) -> SnapshotRecord:
    sha = full_sha(module)
    rel = Path("snapshots") / f"{idx:03d}_{_slug(name)}.mlir"
    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / rel).write_text(text)
    counts = op_counts(module)
    return SnapshotRecord(
        idx=idx, stage=stage, pass_name=name, pass_text=kw.pop("pass_text", name),
        composite=kw.pop("composite", False), changed=prev_sha is not None and sha != prev_sha,
        time_ms=kw.pop("time_ms", 0.0), op_counts=dict(counts), total_ops=sum(counts.values()),
        path=str(rel), sha1=sha, **kw,
    )


def run_mlir_pipeline(
    module: ir.Module,
    config: PipelineConfig,
    out_dir: Path,
    on_pass: Callable[[int, int, str], None] | None = None,
) -> list[SnapshotRecord]:
    """Run ``config`` on ``module`` in place. Returns one record per snapshot."""
    ctx = module.context
    out_dir = Path(out_dir)
    records: list[SnapshotRecord] = []

    text = print_module(module)
    records.append(snapshot(module, out_dir, 0, "torch", "import", text, None,
                            kind="import"))
    prev = records[-1].sha1

    plan: list[tuple[str, str, bool]] = []
    to_leaves = parse_and_print(ctx)
    for stage, spec in config.all_pass_texts():
        try:
            leaves, composite = expand(spec, to_leaves)
        except Exception as e:  # noqa: BLE001 - surface parser message
            raise PipelineError(stage, spec, str(e)) from e
        plan.extend((stage, leaf, composite) for leaf in leaves)

    for n, (stage, leaf, composite) in enumerate(plan, start=1):
        if on_pass:
            on_pass(n, len(plan), pass_name(leaf))
        t0 = time.perf_counter()
        try:
            with ctx:
                PassManager.parse(f"builtin.module({leaf})").run(module.operation)
        except Exception as e:  # noqa: BLE001
            raise PipelineError(stage, leaf, str(e)) from e
        dt = (time.perf_counter() - t0) * 1000
        text = print_module(module)
        rec = snapshot(module, out_dir, n, stage, pass_name(leaf), text, prev,
                       pass_text=leaf, composite=composite, time_ms=dt)
        records.append(rec)
        prev = rec.sha1
    return records


@dataclass
class TailResult:
    records: list[SnapshotRecord] = field(default_factory=list)
    llvm_ir: str = ""
    anchor_path: str = ""


def run_infrastructure_tail(
    module: ir.Module,
    out_dir: Path,
    start_idx: int,
    triple: str | None = None,
) -> TailResult:
    """Prepare the LLVM-dialect module for native build.

    * ``snapshot-op-locations`` rewrites every op location to its line in the anchor
      file (``anchor/llvm_dialect.mlir``), which is what ``!dbg`` later refers to.
    * ``ensure-debug-info-scope-on-llvm-func`` adds the DISubprogram scope.
    * ``translate_module_to_llvmir`` emits textual LLVM IR in-process.
    """
    out_dir = Path(out_dir)
    ctx = module.context
    anchor = out_dir / "anchor" / "llvm_dialect.mlir"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    res = TailResult(anchor_path=str(anchor.relative_to(out_dir)))
    prev = None
    for k, spec in enumerate([
        f"snapshot-op-locations{{filename={anchor}}}",
        "ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}",
    ]):
        t0 = time.perf_counter()
        try:
            with ctx:
                PassManager.parse(f"builtin.module({spec})").run(module.operation)
        except Exception as e:  # noqa: BLE001
            raise PipelineError("llvm_dialect", spec, str(e)) from e
        dt = (time.perf_counter() - t0) * 1000
        text = print_module(module)
        rec = snapshot(module, out_dir, start_idx + k, "llvm_dialect", pass_name(spec), text,
                       prev, pass_text=spec, time_ms=dt, kind="instrumentation")
        res.records.append(rec)
        prev = rec.sha1
    with ctx:
        if triple:
            module.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(triple)
        res.llvm_ir = llvm_d.translate_module_to_llvmir(module.operation)
    return res
