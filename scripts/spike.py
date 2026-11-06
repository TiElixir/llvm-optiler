#!/usr/bin/env python
"""M0 spike: matmul_relu -> Torch -> Linalg -> buffers -> loops -> LLVM dialect -> LLVM IR -> asm -> .so
-> run, check against eager, time both, print a per-pass table for every stage.

    .venv/bin/python scripts/spike.py [template] [--keep DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline.capture import SRC_REL  # noqa: E402
from backend.pipeline.config import PipelineConfig  # noqa: E402
from backend.pipeline.run import compile_run, run_bench_subprocess  # noqa: E402
from backend.toolchain import toolchain_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("template", nargs="?", default="matmul_relu")
    ap.add_argument("--keep", default=str(ROOT / "runs" / "spike"))
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()

    tc = toolchain_report()
    print("toolchain:", json.dumps({k: tc[k] for k in
          ("torch", "torch_mlir", "bundled_mlir_llvm", "system_llvm", "clang", "triple", "host_cpu")}))
    for w in tc["warnings"]:
        print("WARNING:", w)

    run_dir = Path(a.keep)
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "source").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "templates" / f"{a.template}.py", run_dir / SRC_REL)

    t0 = time.perf_counter()
    out = compile_run(run_dir, PipelineConfig.load(), benchmark=False,
                      emit=lambda ev, **k: print(f"[{time.perf_counter() - t0:5.1f}s] {ev} {k or ''}") if ev != "passes" else None)
    recs = out["records"]

    print(f"\n{'idx':>3} {'stage':14} {'pass':44} {'cmp':>3} {'chg':>3} {'ops':>5} {'ms':>8}")
    for r in recs:
        print(f"{r.idx:3d} {r.stage:14} {r.pass_name[:44]:44} {'Y' if r.composite else '.':>3} "
              f"{'Y' if r.changed else '.':>3} {r.total_ops:5d} {r.time_ms:8.1f}")

    b = run_bench_subprocess(run_dir, iters=a.iters)
    print("\ncorrect:", b["correct"], f"max_abs_err={b['max_abs_err']:.3g}")
    if b["correct"]:
        c, e = b["compiled_ms"], b["eager_ms"]
        print(f"compiled: median {c['median']:.4f} ms  (p10 {c['p10']:.4f}, p90 {c['p90']:.4f})")
        print(f"eager   : median {e['median']:.4f} ms  (p10 {e['p10']:.4f}, p90 {e['p90']:.4f})")
        print(f"ratio vs eager: {b['ratio_vs_eager']:.2f}x  alloc: {b['alloc']}")
    return 0 if b["correct"] else 2


if __name__ == "__main__":
    sys.exit(main())
