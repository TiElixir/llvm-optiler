"""Machine calibration for the estimated runtime shares.

``load_or_run`` measures sustained bandwidth and multiply-add throughput once per machine
(cached as JSON, keyed by CPU model and clang version) with small C programs compiled by
clang and pinned to the benchmark core. ``est_line_shares`` and ``est_total_ms`` turn static
op costs (``backend.profile.cost``) into an estimated runtime, used when perf is unavailable.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline.run import bench_core
from backend.toolchain import ToolMissing, cpu_model, find_tool, llvm_version, require_tool

ROOT = Path(__file__).resolve().parents[2]
REPS = 5

_BW_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
  size_t bytes = strtoull(argv[1], 0, 10);
  long inner = strtol(argv[2], 0, 10);
  char *src = malloc(bytes), *dst = malloc(bytes);
  memset(src, 1, bytes); memset(dst, 0, bytes);
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    double t0 = now();
    for (long k = 0; k < inner; ++k) {
      memcpy(dst, src, bytes);
      __asm__ volatile("" : : "r"(dst), "r"(src) : "memory");  /* keep every copy */
    }
    double t = now() - t0;
    if (t < best) best = t;
  }
  printf("%.9g %d\n", best, dst[bytes / 2]);
  return 0;
}
"""

_FMA_C = r"""
#include <stdio.h>
#include <time.h>
#define N 1024
#define ITERS 20000
static float x[N] __attribute__((aligned(64)));
static float y[N] __attribute__((aligned(64)));
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__attribute__((noinline)) static void kernel(float *restrict yy, const float *restrict xx, long iters) {
  for (long it = 0; it < iters; ++it)
    for (int i = 0; i < N; ++i)
      yy[i] = yy[i] * xx[i] + 1.0f;
}
int main(void) {
  for (int i = 0; i < N; ++i) { x[i] = 0.5f + i * 1e-6f; y[i] = 0.0f; }
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    double t0 = now();
    kernel(y, x, ITERS);
    double t = now() - t0;
    if (t < best) best = t;
  }
  double s = 0; for (int i = 0; i < N; ++i) s += y[i];
  printf("%.9g %.6f\n", best, s);
  return 0;
}
"""
FMA_FLOPS = 2.0 * 1024 * 20000  # one mul and one add per element update


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    return r.stdout


def _compile(clang: str, src: str, out: Path, flags: list[str]) -> Path:
    c_file = out.with_suffix(".c")
    c_file.write_text(src)
    _run([clang, *flags, "-o", str(out), str(c_file)])
    return out


def _pinned(exe: list[str], core: int, taskset: str | None) -> list[str]:
    return [taskset, "-c", str(core), *exe] if taskset else exe


def _measure(clang: str) -> dict:
    t_start = time.perf_counter()
    core = bench_core()
    taskset = find_tool("taskset")
    with tempfile.TemporaryDirectory(prefix="optiler-calib-") as tmp:
        d = Path(tmp)
        bw = _compile(clang, _BW_C, d / "bw", ["-O2"])
        fs = _compile(clang, _FMA_C, d / "fs", ["-O1", "-fno-vectorize", "-fno-slp-vectorize"])
        fv = _compile(clang, _FMA_C, d / "fv", ["-O3", "-march=native"])

        def bw_gbs(size: int, inner: int) -> float:
            best, _ = map(float, _run(_pinned([str(bw), str(size), str(inner)], core, taskset)).split()[:2])
            return 2.0 * size * inner / best / 1e9  # bytes read + bytes written, as in cost.py

        def gflops(exe: Path) -> float:
            best, _ = map(float, _run(_pinned([str(exe)], core, taskset)).split()[:2])
            return FMA_FLOPS / best / 1e9

        bandwidth = bw_gbs(64 << 20, 2)
        bandwidth_cache = bw_gbs(256 << 10, 1024)
        scalar = gflops(fs)
        vector = gflops(fv)
    return {
        "bandwidth_gbs": bandwidth,
        "bandwidth_cache_gbs": bandwidth_cache,
        "scalar_gflops": scalar,
        "vector_gflops": vector,
        "seconds": time.perf_counter() - t_start,
    }


def _cache_path(cache_path: Path | None) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    return Path(os.environ.get("OPTILER_RUNS_DIR", ROOT / "runs")) / "calibration.json"


def load_or_run(cache_path: Path | None = None, force: bool = False) -> dict:
    """Calibration numbers for this machine; measured once and cached as JSON.

    A cache written for a different CPU model or clang version is measured again.
    Raises ``ToolMissing`` if clang is not installed.
    """
    clang = require_tool("clang")
    cpu = cpu_model()
    clang_ver = llvm_version("clang") or "unknown"
    path = _cache_path(cache_path)
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text())
            if cached.get("cpu") == cpu and cached.get("clang") == clang_ver:
                return cached
        except (OSError, ValueError):
            pass
    measured = _measure(clang)
    result = {**measured, "cpu": cpu, "clang": clang_ver,
              "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _op_seconds(cost: dict[str, float], calib: dict) -> float:
    """Estimated time of one op: the slower of its compute and its memory traffic.

    ``max(flops / (scalar_gflops * 1e9), bytes / (bandwidth_gbs * 1e9))``. The scalar rate is
    used because the default pipeline emits unvectorized loops.
    """
    compute = cost.get("flops", 0.0) / (calib["scalar_gflops"] * 1e9)
    memory = cost.get("bytes", 0.0) / (calib["bandwidth_gbs"] * 1e9)
    return max(compute, memory)


def est_line_shares(ops: list[tuple[list[int], dict[str, float]]], calib: dict) -> dict[int, float]:
    """Estimated share of runtime per source line.

    ``ops`` holds ``(source_lines, cost)`` pairs. Each op's time (see ``_op_seconds``) is split
    equally across its distinct lines. Ops without lines are ignored, so the shares sum to 1.0
    over the attributed time. Returns ``{}`` when the total is zero.
    """
    per_line: dict[int, float] = {}
    total = 0.0
    for lines, cost in ops:
        if not lines:
            continue
        uniq = sorted(set(lines))
        t = _op_seconds(cost, calib)
        total += t
        for ln in uniq:
            per_line[ln] = per_line.get(ln, 0.0) + t / len(uniq)
    if total <= 0.0:
        return {}
    return {ln: t / total for ln, t in per_line.items()}


def est_total_ms(ops: list[tuple[list[int], dict[str, float]]], calib: dict) -> float:
    """Estimated runtime in ms: the sum of ``_op_seconds`` over all given ops."""
    return 1000.0 * sum(_op_seconds(cost, calib) for _lines, cost in ops)
