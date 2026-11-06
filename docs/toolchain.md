# Toolchain (pinned and verified on 2026-11-07)

| Component | Version | Where from |
|---|---|---|
| Python | 3.12.14 | `uv venv --python 3.12 .venv` |
| torch | 2.15.0+cpu | stable CPU wheel (torch-mlir main pins 2.15.0.dev20260917, same series) |
| torch-mlir | 20260930.892 (cp312-abi3 manylinux wheel) | https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels |
| MLIR inside torch-mlir | LLVM 24.0.0git | `torch-mlir-opt --version` |
| clang / opt / llc / llvm-objdump / llvm-symbolizer | LLVM 23.1.5 release tarball | `scripts/setup_llvm.sh` → `toolchain/llvm/bin` |
| perf | 7.2.9 (system) | `perf_event_paranoid=2`, user-space sampling works |
| Host | Linux x86_64, i7-11800H (Tiger Lake), AVX-512 | `llc` host CPU = `tigerlake` |

`backend/toolchain.py` looks for tools in `$OPTILER_LLVM_BIN`, then `toolchain/llvm/bin`, then `PATH`.

**Why a local tarball.** Arch's `llvm` package is 23.1.1 but a partial upgrade (new `llvm`, old
`llvm-libs`) leaves `opt` unable to load `libLLVM.so.23.1`, and a full `-Syu` was not wanted.
The tarball leaves the system untouched.

## Facts the design depends on (answers to the M0 questions)

1. **Version skew works (so far).** torch-mlir's MLIR is LLVM 24.0.0git, `opt`/`llc` are 23.1.5.
   Textual LLVM IR from `translate_module_to_llvmir` is accepted by `opt` 23.1.5 for `matmul_relu`
   and the `nn.Linear` test net. `/api/toolchain` warns about the major-version difference. If a
   future model breaks this, pin an older torch-mlir nightly from the LLVM-23 era.
2. **Pass names.** `torch-mlir-opt` registers only the torch passes. Upstream MLIR passes
   (bufferization, linalg, affine, scf, memref, LLVM conversions, transform) are available through the
   Python bindings (`PassManager.parse`). `scripts/check_passes.py` is the Rule 1 gate (also run from pytest).
3. **Composites expand.** `str(PassManager.parse("builtin.module(<composite>)"))` prints the component
   passes, so `torchdynamo-export-to-torch-backend-pipeline` (6 passes),
   `torch-backend-to-linalg-on-tensors-backend-pipeline` (18) and `buffer-deallocation-pipeline` (7)
   are run one pass at a time. Each leaf is marked `composite: true` in the UI.
4. **Locations.** FX import gives `loc("source/model.py":LINE:0)`, line-level only (several FX nodes can
   share a line). `nn.Linear` produces `callsite(<torch/nn/modules/linear.py> at <source/model.py:LINE>)`,
   so the lineage root is the *caller* of the callsite. About 94% of ops keep a known location down to the
   LLVM dialect (`matmul_relu`).
5. **Debug info chain.** `snapshot-op-locations{filename=anchor/llvm_dialect.mlir}` rewrites every op
   location to its line in that file; `ensure-debug-info-scope-on-llvm-func` adds the DISubprogram;
   `!DILocation(line: N)` in the LLVM IR, `DebugLoc: { File: llvm_dialect.mlir, Line: N }` in optimization
   remarks and `; .../anchor/llvm_dialect.mlir:N` line markers from `llvm-objdump -l` all refer to anchor
   lines. Original locations are replaced by that pass, so provenance clones the module first and walks
   both copies in the same order.
6. **`-print-changed=quiet -print-module-scope`** prints `*** IR Dump After <Pass> on <function|[module]> ***`
   followed by the whole module, only for passes that changed the IR (46 of them for `matmul_relu`).
7. **C interface.** `func.func(llvm-request-c-wrappers)` makes `convert-to-llvm` emit
   `_mlir_ciface_forward(result*, arg0*, ...)`. The ctypes harness calls that.
8. **Allocation.** `finalize-memref-to-llvm{use-generic-functions=true}` lowers to
   `_mlir_memref_to_llvm_alloc/free`, implemented in `backend/runtime/optiler_rt.c`, which counts
   allocations and live/peak bytes.
9. **Large constants.** `nn.Linear` weights become `dense_resource` blobs (660 KB hex per snapshot for a
   256x256 layer). Stored snapshots are printed with `large_elements_limit=16, large_resource_limit=16`
   (elided, display only); `changed` hashes the full text. LLVM-stage snapshots elide long `@global`
   initializers; the real IR for `opt` lives in `llvm/full/`.

## First measurement (M0)

`matmul_relu` 256x256x256, 1 thread pinned to core 0, 100 iterations:
compiled 28.1 ms (p10 27.0, p90 29.2) vs eager 0.66 ms, correct (max abs err 1.5e-5).
The matmul reduction loop is not vectorized: `loop not vectorized: cannot prove it is safe to reorder
floating-point operations` (remark at anchor line 96) and the asm has scalar `vmulss`/`vaddss` only. This
is the situation detector D3 exists for.

## M0b: pipeline patches on `matmul_relu` (measured, 50 iterations, core-pinned, `scripts/viability.py`)

| variant | median ms | vs baseline | correct |
|---|---|---|---|
| baseline (default pipeline) | 30.6 | 1.00x | yes |
| drop `affine-loop-fusion` | 30.1 | 1.02x | yes |
| drop `linalg-fuse-elementwise-ops` | 28.7 | 1.07x | yes |
| transform `interchange [0,2,1]` + drop `affine-loop-fusion` | **2.25** | **13.6x** | yes |

(eager PyTorch, 1 thread: 0.64–0.82 ms.)

Why: `affine-loop-fusion` fuses the zero-fill, the matmul and the bias+relu into one `(i, j)` nest with the
reduction loop `k` innermost, which LLVM cannot vectorize without reassociating floating point. The interchange
alone is undone by that fusion. With `k` moved out and fusion off, the innermost loop `j` is parallel and
contiguous, and `loop-vectorize` accepts it. So the first fix D3 can offer is measured to work.
Transform scripts are attached with `transform-preload-library{transform-library-paths=${RUN_DIR}/transforms/t.mlir}`
followed by `transform-interpreter` at the end of `linalg_tensor`; `${RUN_DIR}` is substituted at run time.

### M0b across all templates (median ms, 50 iterations, core-pinned; eager is PyTorch at 1 thread)

| template | baseline | drop affine fusion | **interchange + drop affine fusion** | drop elementwise fusion | eager |
|---|---|---|---|---|---|
| attention_block | 6.74 | 4.43 (1.52x) | **0.518 (13.0x)** | 4.28 (1.58x) | 0.23 |
| mlp | 3.35 | 2.07 (1.62x) | **0.217 (15.4x)** | 4.40 (0.76x) | 0.11 |
| softmax | 1.04 | 1.01 | 1.03 | **wrong output** | 0.20 |
| layernorm | 0.75 | 0.74 | 0.77 (0.96x) | 1.16 (0.64x) | 0.09 |

* Matmul-bearing models (`attention_block`, `mlp`, `matmul_relu`) get a 13-15x speedup from the interchange
  fix, so the M6 acceptance check (a child run that is measurably faster) is reachable with a real,
  verified fix. The compiled code is still 2-4x slower than eager (oneDNN/MKL), which is expected.
* Models without a matmul (`softmax`, `layernorm`) are not helped by it; no fix is offered for them
  unless a detector finds a real cause.
* Removing `linalg-fuse-elementwise-ops` from `softmax` produces wrong results (correctness check fails),
  and slows `mlp`/`layernorm`. This is why speed claims are hidden whenever the correctness check fails.
* The demo narrative "unfused softmax" is therefore a hypothesis (decisions.md #7). The measured lead
  finding on `attention_block` is the unvectorized matmul reduction loop (D3).
