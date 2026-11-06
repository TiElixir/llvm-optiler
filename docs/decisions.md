# Decisions and deviations from PLAN.md

PLAN.md stays the product spec. This file records where the build differs from it, and why.

1. **Attribution is `perf | probe | est`.** PLAN.md 8.2 has only `perf | est`. Probe mode builds a second, instrumented `model_probe.so` with a cycle-counter timer around every top-level loop nest, so measured per-op shares exist on machines without `perf`. The headline runtime always comes from the uninstrumented build.
2. **Extra dependencies (approved by the user):** `httpx` (FastAPI TestClient), `vitest`, `@playwright/test`. Fonts are asked about separately in M2.
3. **Native build:** `opt -passes='default<O3>'`, then `llc -O3`, then `clang -shared` (link only). O3 runs once, and the `asm` stage is the disassembly of the exact `.so` that runs.
4. **Measured allocations.** The MLIR memref alloc/free helpers are implemented in `backend/runtime/optiler_rt.c`, so `alloc_count` and `peak_bytes` are measured. The static liveness walk (8.3) remains as the `est` value per stage.
5. **"Measurably faster"** means child p90 < parent p10 and ratio >= 1.05. Otherwise a finding shows "no measurable change", or "slower xN" when the child is slower.
6. **Tailwind v4.** Tokens live in `tokens.css` and are mapped with `@theme inline`; this plays the role of `tailwind.config` in PLAN.md 4.4. Components never use raw hex.
7. **The demo narrative is a hypothesis.** If measurement on `attention_block` shows a different top finding than "unfused softmax", the demo text or template dims change to fit the data, never the reverse (Rule 2).
8. **Python 3.12** (venv via `uv`). The system Python is 3.14, which torch-mlir wheels do not support.
9. **Own git repo** in `MLIR_v2/`, at most 12 planned commits, one per milestone after its checks pass.
