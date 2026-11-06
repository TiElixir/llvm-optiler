"""One compile job: ``python -m backend.runner.job --runs-dir DIR RUN_ID``.

Runs in its own process so a crashing MLIR pass cannot take the API or the worker down.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from backend.pipeline.capture import CaptureError
from backend.pipeline.config import STAGES, PipelineConfig
from backend.pipeline.llvm_steps import BuildError
from backend.pipeline.mlir_runner import PipelineError, SnapshotRecord
from backend.pipeline.run import compile_run
from backend.schemas import FxNode, Metrics, ProgressEvent, Run, Snapshot, Stage
from backend.store.runs import RunStore
from backend.toolchain import ToolMissing, toolchain_report

STAGE_NAMES = {
    "source": "Source / FX", "torch": "Torch dialect", "linalg_tensor": "Linalg (tensors)",
    "linalg_buffer": "Linalg (buffers)", "loops": "Loops (SCF/Affine)",
    "llvm_dialect": "LLVM dialect", "llvm_ir": "LLVM IR", "asm": "Assembly",
}


def build_stages(snapshots: list[Snapshot]) -> list[Stage]:
    return [Stage(id=s, name=STAGE_NAMES[s],
                  snapshot_idxs=[x.idx for x in snapshots if x.stage == s]) for s in STAGES]


def snapshot_model(r: SnapshotRecord) -> Snapshot:
    return Snapshot(idx=r.idx, stage=r.stage, **{"pass": r.pass_name}, pass_text=r.pass_text,
                    composite=r.composite, changed=r.changed, time_ms=r.time_ms,
                    op_counts=r.op_counts, total_ops=r.total_ops, path=r.path, kind=r.kind)


def metrics_from_bench(b: dict, run: Run) -> Metrics:
    m = Metrics(threads=run.settings.threads, iters=run.settings.iters,
                correct=b["correct"], max_abs_err=b["max_abs_err"], rtol=b["rtol"], atol=b["atol"])
    if not b["correct"]:
        return m  # a wrong result hides every speed claim
    c, e = b["compiled_ms"], b["eager_ms"]
    m.runtime_ms_median, m.runtime_ms_p10, m.runtime_ms_p90 = c["median"], c["p10"], c["p90"]
    m.eager_ms_median, m.eager_ms_p10, m.eager_ms_p90 = e["median"], e["p10"], e["p90"]
    m.ratio_vs_eager = b["ratio_vs_eager"]
    m.peak_bytes, m.alloc_count = b["alloc"]["peak_bytes"], b["alloc"]["alloc_count"]
    return m


def analyze(store: RunStore, run: Run, out: dict) -> None:
    """Hook for provenance (M3), attribution (M4) and detectors (M5)."""


def run_job(store: RunStore, run_id: str) -> None:
    run = store.get(run_id)
    run.status, run.phase = "running", "capture"
    store.save(run)
    d = store.run_dir(run_id)

    def emit(ev: str, **k) -> None:
        phase = {"capture": "capture", "import": "import", "passes": "passes",
                 "build": "build", "benchmark": "benchmark"}.get(ev, ev)
        name = k.get("name") or k.get("step")
        store.append_event(run_id, ProgressEvent(phase=phase, n=k.get("n"), total=k.get("total"), name=name))
        if run.phase != phase:
            run.phase = phase
            store.save(run)

    try:
        cfg = PipelineConfig.from_yaml(run.pipeline_yaml)
        parent_dir = None
        if run.parent_id:
            parent = store.get(run.parent_id)
            if parent.source_code == run.source_code and (store.run_dir(parent.id) / "capture").exists():
                parent_dir = store.run_dir(parent.id)  # same model: reuse the captured graph
        out = compile_run(d, cfg, source=run.source_code, capture_from=parent_dir,
                          iters=run.settings.iters, warmup=run.settings.warmup,
                          threads=run.settings.threads, emit=emit)
        meta = out["meta"]
        run.inputs_spec, run.output_spec = meta["inputs"], meta["output"]
        run.snapshots = [snapshot_model(r) for r in out["records"]]
        run.stages = build_stages(run.snapshots)
        for st in run.stages:
            so = out["provenance"].get(st.id)
            st.exact_lineage_pct = so.exact_lineage_pct if so else None
        run.fx_nodes = [FxNode(**n) for n in json.loads((d / "capture" / "fx_nodes.json").read_text())]
        run.metrics = metrics_from_bench(out["bench"], run)
        run.toolchain = {k: v for k, v in toolchain_report().items() if k not in ("tools",)}
        emit("analyze")
        analyze(store, run, out)
        run.status, run.phase, run.error = "done", "done", None
        store.save(run)
        store.append_event(run_id, ProgressEvent(type="done", phase="done"))
    except Exception as e:  # noqa: BLE001 - every failure becomes a run error
        run.status, run.phase = "error", "error"
        if isinstance(e, PipelineError):
            run.error, run.error_stage = e.message, e.stage
        elif isinstance(e, CaptureError):
            run.error, run.error_stage = str(e), "source"
        elif isinstance(e, BuildError):
            run.error, run.error_stage = f"{e.step}: {e.output[-1500:]}", "llvm_ir"
        elif isinstance(e, ToolMissing):
            run.error, run.error_stage = str(e), "llvm_ir"
        else:
            run.error = "".join(traceback.format_exception_only(type(e), e)).strip()[-1500:]
            sys.stderr.write(traceback.format_exc())
        store.save(run)
        store.append_event(run_id, ProgressEvent(type="error", phase="error", message=run.error))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("run_id")
    a = ap.parse_args()
    run_job(RunStore(Path(a.runs_dir)), a.run_id)


if __name__ == "__main__":
    main()
