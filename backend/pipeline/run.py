"""compile_run: the whole Compile step for one run directory.

capture (sandboxed subprocess) -> MLIR passes one at a time -> LLVM-dialect tail ->
opt/llc/link/objdump -> pinned benchmark subprocess. Progress goes through ``emit``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

from backend.toolchain import find_tool, target_triple

from .capture import SRC_REL, run_capture
from .config import PipelineConfig, StageSpec, PassSpec
from .llvm_steps import BuildResult, build_native
from .mlir_runner import SnapshotRecord, TailResult, run_infrastructure_tail, run_mlir_pipeline

ROOT = Path(__file__).resolve().parents[2]
Emit = Callable[..., None]


def bench_core() -> int:
    """Core the benchmark is pinned to (override with OPTILER_BENCH_CORE)."""
    env = os.environ.get("OPTILER_BENCH_CORE")
    if env is not None:
        return int(env)
    cores = sorted(os.sched_getaffinity(0))
    return cores[min(3, len(cores) - 1)]


def substitute_run_dir(config: PipelineConfig, run_dir: Path) -> PipelineConfig:
    """Replace ``${RUN_DIR}`` in pass texts (transform script paths) with the real directory."""
    d = str(Path(run_dir).resolve())
    stages = []
    for s in config.stages:
        passes = []
        for p in s.passes:
            q = p.model_copy(deep=True)
            q.pass_ = q.pass_.replace("${RUN_DIR}", d)
            if q.options:
                q.options = {k: str(v).replace("${RUN_DIR}", d) for k, v in q.options.items()}
            passes.append(q)
        stages.append(StageSpec(stage=s.stage, passes=passes))
    return PipelineConfig(stages=stages)


def run_bench_subprocess(run_dir: Path, iters: int = 100, warmup: int = 10, threads: int = 1,
                         so: str = "bin/model.so", out: str = "reports/bench.json") -> dict:
    cmd = [sys.executable, "-P", "-m", "backend.execute.bench", str(run_dir.resolve()),
           "--iters", str(iters), "--warmup", str(warmup), "--threads", str(threads),
           "--so", so, "--out", out]
    taskset = find_tool("taskset")
    if taskset:
        cmd = [taskset, "-c", str(bench_core())] + cmd
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads)}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("benchmark failed:\n" + (r.stderr or r.stdout)[-2000:])
    return json.loads((run_dir / out).read_text())


def compile_run(run_dir: Path, config: PipelineConfig, *, source: str | None = None,
                capture_from: Path | None = None, benchmark: bool = True,
                iters: int = 100, threads: int = 1, emit: Emit | None = None) -> dict:
    """Run the full pipeline in ``run_dir``. Returns {records, bench, build, tail}."""
    run_dir = Path(run_dir)
    emit = emit or (lambda *a, **k: None)
    run_dir.mkdir(parents=True, exist_ok=True)

    # capture
    emit("capture")
    if capture_from is not None:  # child runs reuse the parent's captured model
        if (run_dir / "capture").exists():
            shutil.rmtree(run_dir / "capture")
        shutil.copytree(Path(capture_from) / "capture", run_dir / "capture")
        (run_dir / "source").mkdir(exist_ok=True)
        shutil.copy(Path(capture_from) / SRC_REL, run_dir / SRC_REL)
        meta = json.loads((run_dir / "capture" / "meta.json").read_text())
    else:
        if source is not None:
            (run_dir / "source").mkdir(exist_ok=True)
            (run_dir / SRC_REL).write_text(source)
        meta = run_capture(run_dir)

    # MLIR passes
    emit("import")
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    with ctx:
        module = ir.Module.parse((run_dir / "capture" / "import.mlir").read_text())
    resolved = substitute_run_dir(config, run_dir)
    records: list[SnapshotRecord] = run_mlir_pipeline(
        module, resolved, run_dir, on_pass=lambda n, total, name: emit("passes", n=n, total=total, name=name))
    tail: TailResult = run_infrastructure_tail(module, run_dir, len(records), triple=target_triple())
    records += tail.records

    # native build
    emit("build")
    build: BuildResult = build_native(run_dir, tail.llvm_ir, len(records),
                                      on_step=lambda s: emit("build", step=s))
    records += build.records

    bench = None
    if benchmark:
        emit("benchmark")
        bench = run_bench_subprocess(run_dir, iters=iters, threads=threads)
    return {"records": records, "bench": bench, "build": build, "tail": tail, "meta": meta}
