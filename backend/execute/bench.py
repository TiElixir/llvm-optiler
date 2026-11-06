"""Benchmark + correctness child process.

Run as ``taskset -c <core> python -m backend.execute.bench <run_dir> [opts]`` so timings
are pinned to one core and no other compile job overlaps (single worker).
Writes ``<run_dir>/reports/bench.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q))


def time_calls(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def summarize(ms: list[float]) -> dict:
    return {"median": _pct(ms, 50), "p10": _pct(ms, 10), "p90": _pct(ms, 90), "n": len(ms)}


def run_bench(run_dir: Path, so_name: str = "bin/model.so", warmup: int = 10, iters: int = 100,
              threads: int = 1, rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    import torch

    from .harness import CompiledModel

    torch.set_num_threads(threads)
    meta = json.loads((run_dir / "capture" / "meta.json").read_text())
    z = np.load(run_dir / "capture" / "inputs.npz")
    arrays = [z[f"arr_{i}"] for i in range(len(meta["inputs"]))]
    ref = np.load(run_dir / "capture" / "eager_out.npz")["arr_0"]

    cm = CompiledModel(run_dir / so_name, len(arrays), tuple(meta["output"]["shape"]))
    got = cm(arrays)
    ok = bool(np.allclose(got, ref, rtol=rtol, atol=atol)) and got.shape == ref.shape
    max_err = float(np.max(np.abs(got - ref))) if got.shape == ref.shape else float("inf")
    result: dict = {"correct": ok, "max_abs_err": max_err, "rtol": rtol, "atol": atol,
                    "threads": threads, "pinned_core": sorted(os.sched_getaffinity(0))}
    if not ok:
        return result

    # single-iteration allocation profile (measured)
    keep, _d, ptrs = cm.prepare(arrays)
    cm.reset_stats()
    res = cm.call_raw(ptrs)
    cm.release(res)
    result["alloc"] = cm.alloc_stats()

    def compiled_once():
        r = cm.call_raw(ptrs)
        cm.release(r)

    ms = time_calls(compiled_once, warmup, iters)
    result["compiled_ms"] = summarize(ms)

    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir / "source"))
    ns = runpy.run_path("source/model.py")
    model = ns["model"]
    if hasattr(model, "eval"):
        model.eval()
    tin = tuple(torch.from_numpy(a.copy()) for a in arrays)
    with torch.no_grad():
        ems = time_calls(lambda: model(*tin), warmup, iters)
    result["eager_ms"] = summarize(ems)
    result["ratio_vs_eager"] = result["compiled_ms"]["median"] / result["eager_ms"]["median"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--so", default="bin/model.so")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="reports/bench.json")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).resolve()
    res = run_bench(run_dir, a.so, a.warmup, a.iters, a.threads)
    out = run_dir / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
