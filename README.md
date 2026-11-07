# Optiler

Paste a small PyTorch model. Optiler lowers it through a real MLIR pipeline to native code, runs it against PyTorch eager, and tells you why it is slow and how to fix it, with the fix proven by a measured before/after.

```
compile -> diagnose -> fix -> compare
```

1. **Compile.** Torch FX -> Torch dialect -> Linalg on tensors -> buffers -> loops -> LLVM dialect -> LLVM IR -> assembly. Every pass is a snapshot; the result is built, run, checked against eager and benchmarked (pinned core, median and p10/p90).
2. **Diagnose.** The *pipeline map* traces any source line through every stage down to the assembly (exact lineage, inferred links drawn dashed, coverage per stage). Runtime is attributed to lines with `perf` (or a labeled static estimate). Detectors report findings only when certain: unfused elementwise chains (D1), temporaries that get copied (D2), hot matmul loops LLVM could not vectorize (D3).
3. **Fix.** A fix is a pipeline patch. Applying it creates a *child run*; nothing is overwritten.
4. **Compare.** The child's measured result marks the finding `verified` (non-overlapping timing ranges and at least 1.05x), or says plainly that nothing changed or it got slower. Compare shows metrics, op counts per stage, the pipeline diff and the IR diff per stage.

Numbers are either measured or labeled `est`. If the output does not match eager, every speed claim is hidden.

## Setup (Linux x86_64)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
  "torch==2.15.0+cpu" "torch-mlir==20260930.892" numpy fastapi uvicorn "pydantic>=2" pyyaml sse-starlette zstandard pytest httpx
scripts/setup_llvm.sh      # project-local LLVM 23.1.5 in toolchain/llvm (~1.2 GB download, system untouched)
```

`perf` is optional (without it, runtime shares are static estimates and say so). Versions and why they are pinned: [docs/toolchain.md](docs/toolchain.md).

## Run

```bash
.venv/bin/uvicorn backend.api.app:app --port 8000     # API + UI + one compile worker
```

Open http://127.0.0.1:8000. `ctrl+enter` compile, `j`/`k` findings, `f` apply the fix, `[` `]` move between stages, `esc` clear. The **advanced** drawer has settings, an editable pipeline (checked against the toolchain before it is queued), a transform-dialect script, every pass with timings, and a reproducer zip.

The UI is plain HTML/CSS/JS served by the API: no build step, no npm dependencies. The thinking orbs shown while compiling are [thinking-orbs](https://github.com/RareFormLabs/thinking-orbs) (MIT), vendored in `frontend/vendor`.

## Tests

```bash
.venv/bin/python -m pytest backend/tests -q            # ~2.5 min: compiles and benchmarks the templates
node --test frontend/test                              # pure JS, instant
NODE_PATH=/usr/lib/node_modules node frontend/e2e/smoke.cjs   # whole loop in a browser (global Playwright + system Chrome)
.venv/bin/python scripts/check_passes.py               # every pass name exists in the pinned toolchain
```

## Layout

```
backend/pipeline     capture (sandboxed), per-pass MLIR runner, LLVM steps, compile_run
backend/provenance   op tables, location parsing, lineage through LLVM IR and asm
backend/profile      perf attribution, static cost model, calibration
backend/detectors    D1-D3        backend/fixes   fix registry, patch, verdict
backend/runner       one worker, one job process per run
backend/api          routes, SSE, reproducer       backend/templates   five demo models
frontend             static UI     docs   toolchain, provenance, decisions (deviations from PLAN.md)
```

PLAN.md is the original spec; `docs/decisions.md` lists where the build differs and why.

## Limits

GPU targets, training graphs, Hugging Face loading, accounts, cloud deployment, autotuning and multi-threaded codegen are out of scope. Models must return one float32 tensor. Lineage is per source line (the finest identity the importer's locations carry). The estimated speedups are conservative by design (the D3 estimate for `attention_block` is about 2x; the measured fix is about 12x).
