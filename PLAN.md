# PLAN.md — Optiler v2 (working name)

> This file is the build prompt for the coding agent. Read it fully before writing code.
> Build **one milestone at a time** (Section 12). Do not add features that are not in this file.

---

## 0. Rules for the agent

1. **Never invent MLIR/LLVM pass names or flags.** Every pass in the pipeline config must be verified against the pinned toolchain (`--help` output or the Python bindings). If a pass named here does not exist in the pinned version, find the equivalent, update the config, and note it in `docs/toolchain.md`.
2. **Real data only.** Every number shown in the UI is either *measured* or explicitly labeled *estimate*. No mock data in the shipped app (fixtures are fine in tests).
3. **The original compiler output is the source of truth.** Parsed views and summaries are navigation aids; the raw IR is always one click away.
4. **Minimal surface area.** If a feature does not serve the core loop (Section 2), it goes in the Advanced drawer or does not get built.
5. **Ask before adding a dependency** not listed in Section 11b.
6. Each milestone ends with its acceptance checks passing (Section 12). Do not start the next one before that.

---

## 1. What we are building

A tool that answers one question for an ML model compiled through MLIR:

> **Why is this model slow, and how do I fix it?**

The user pastes a small PyTorch model. The tool lowers it through a real multi-stage pipeline — **Torch FX → Torch dialect → Linalg on tensors → bufferized Linalg → loops (SCF/Affine) → LLVM dialect → LLVM IR → assembly** — runs the resulting native code, compares it to PyTorch eager, explains where time and memory go, points to the exact op and pass responsible, offers a fix, applies it, and proves the result with a measured before/after.

### Context
- Source problem: SegFault 2026 "AI Compiler Optimization Explorer". The previous version (v1) stopped at Linalg-on-tensors and was judged insufficient.
- Prior art to beat: **CompilerLens** (IREE-based; stage capture to assembly, op lineage, option playground, benchmarking) and **LLVM Lens** (LLVM pass blame view).
- **Our differentiator:** not just *seeing* the pipeline, but a closed **Compile → Diagnose → Fix → Compare** loop with evidence-backed findings and verified speedups. Visualization exists to support that loop.

### Target users
Senior compiler engineers and ML systems researchers. They read IR fluently, distrust tools that guess, and will drop the tool after one wrong diagnosis. Precision beats breadth.

---

## 2. The core loop (the whole product)

1. **Compile** — paste model (or pick a template) → one button. Full lowering, per-pass snapshots, native build, benchmark vs eager, correctness check. No configuration required.
2. **Diagnose** — a headline number plus a ranked list of findings, each tied to exact ops, stages and passes.
3. **Fix** — each finding offers a concrete fix (a pipeline change). Applying it creates a **new child run**; nothing is overwritten.
4. **Compare** — parent vs child: runtime, memory, op counts, IR diff. The finding is marked *verified* with the measured effect.

Everything in the UI starts from an op, a finding, or a run. No standalone menus or tabs.

---

## 3. Screens and interactions

There is **one main screen** (the Workspace) plus an empty/start state.

### 3.1 Start state
- Centered, minimal: product wordmark, one-line tagline, the code editor with a template preloaded, a template picker, and a single `compile` button.
- The only place the decorative background (Section 4.4) appears.

### 3.2 Workspace layout (desktop, ≥1280px)

```
┌──────────────────────────────────────────────────────────────────────┐
│ HEADLINE BAR                                                          │
│ 3.21 ms  ·  1.8× slower than eager (1 thread)  ·  64% in softmax:L14  │
│ run history:  #1 ─ #2 (fuse softmax) ─ #3 (tile 32)      [advanced]   │
├───────────────┬──────────────────────────────────────────────────────┤
│ FINDINGS      │ PIPELINE MAP                                          │
│ 01 ▲ 1.6× est │  source   L12 linear   L13 add   L14 softmax          │
│ unfused ...   │  torch    ...                                         │
│ 02 ▲ 1.2× est │  linalg   ...                                         │
│ copy in ...   │  buffers  ...                                         │
│ 03 ...        │  loops    ...                                         │
│               │  llvm     ...                                         │
│               │  asm      ...                                         │
├───────────────┴──────────────────────────────────────────────────────┤
│ INSPECTOR (opens on selection): IR at selected stage | diff vs parent │
└──────────────────────────────────────────────────────────────────────┘
```

**Headline bar**
- Primary metric: median runtime of compiled code. Secondary: ratio vs PyTorch eager under the same thread count (default 1 thread for both; stated in the headline). Tertiary: the hottest source line and its share of runtime.
- If the correctness check fails, the headline turns into an error state: `output mismatch at stage <X> (max abs err …)` and all speed claims are hidden.
- Run history: a horizontal chain of runs. Each node shows run number, fix label, runtime. Clicking selects that run; shift-click a second run opens Compare.

**Findings list (left)**
- Ranked by estimated time impact. Each item: index (`01`), title, one-line evidence, impact (`est 1.6×` or `verified 1.7×`), and a `fix` button.
- Selecting a finding highlights its ops in the Pipeline Map and opens its evidence in the Inspector.
- Empty state: `no findings above threshold` — never pad the list.

**Pipeline Map (center) — the signature view**
- Rows = stages (source/FX, torch, linalg-tensor, linalg-buffer, loops, llvm-dialect, llvm-ir, asm). Each row shows that stage's ops as compact chips in program order, grouped by origin source line.
- Chip color intensity = share of runtime (measured where attribution exists, otherwise hollow chip + "est").
- Clicking any chip (or a source line) highlights its full **lineage** across all stages: ancestors and descendants. Approximate links are drawn dashed and labeled.
- Between rows, a thin pass strip shows which passes ran for that stage transition; passes that changed the IR are marked; hover shows op-count delta (e.g. `linalg.generic 6 → 3`).

**Inspector (bottom, collapsible)**
- Tabs: `ir` (raw IR of the selected stage with the selected ops highlighted), `diff` (selected pass before/after, or run vs parent), `evidence` (for findings: the exact facts and the tool output they came from, e.g. bufferization conflict report or LLVM missed-optimization remark).

**Compare mode**
- Two runs side by side: runtime, peak memory, allocations, kernel count, op counts per stage; IR diff per stage; the pipeline config diff that caused it.

**Advanced drawer** (hidden by default, opened from the headline bar)
- Editable pipeline config (YAML) → `compile` creates a new run.
- Transform-dialect script editor (attach to a run).
- Per-pass raw snapshots list with timings.
- Export reproducer: zip with input `.mlir`, exact commands, pipeline config.
- Settings: benchmark iterations, thread count.

### 3.3 Keyboard
`⌘/Ctrl+Enter` compile · `j/k` move through findings · `f` apply fix for selected finding · `[` `]` move selection up/down a stage · `Esc` clear selection.

---

## 4. Visual design

Reference: **https://segfault.compilertech.org/** — a stark, technical, terminal-flavored aesthetic.

### 4.1 First step: measure the reference
Before building UI, open the reference site in a browser's devtools and record the actual values into `frontend/src/styles/tokens.css`: background and text colors, accent color(s), border color, font families (display, body, mono), base font size, heading sizes, letter-spacing on uppercase labels, border radius, spacing scale. Write the measured values and where they came from in a comment at the top of the file. The defaults below are **placeholders only** until replaced by measured values.

### 4.2 Traits to reproduce
Observed on the reference site (confirmed from a screenshot of its "The format" section):
- **Two type families.** Headlines and body text use a **light-weight geometric sans** (thin strokes, weight ~300–400, large, tight line height, centered for section heroes), often split over two lines ("Everything is online. / Except the finale."). **Monospace is only for small labels**: uppercase, small (~11px), wide letter-spacing.
- **Warm near-black background** (slightly brown, not neutral or blue), **cream/off-white** headlines, warm-gray body text, dimmer gray for indices.
- **Bordered grid ("bento") layout:** content sits in cells separated by 1px warm-gray rules, no gaps, no shadows, no rounded cards. Each cell: mono kicker label top-left (`ANYWHERE`), mono index top-right (`001`), then a sans heading and short gray body text.
- **Kicker pill** above a section: small mono uppercase text with a tiny icon, inside a thin bordered rectangle (`THE FORMAT`).
- **Line-art diagrams:** small bordered boxes with mono uppercase labels, dashed boxes for the central node, thin orthogonal connector lines, corner-bracket marks framing the diagram.
- **One saturated accent:** electric violet/indigo, used as a solid fill behind a single mono label (`FINALE · IISc`) to mark the one important thing. Used very rarely.
- **Numbered indices** as a structural device (`01`, `02`, `001`).
- ASCII-art imagery and a glitch-style wordmark.
- A faint, repeating field of compiler tokens (`fn pass ir ssa cfg … llvm opt loop { } =>`) used as background texture.
- A ticker strip of keywords separated by `·`.
- Tag/pill lists for categories; thin rules and borders instead of shadows or heavy cards.

### 4.3 Translating it into a work tool
This is a tool people stare at for an hour, not a landing page. Apply the aesthetic with restraint:
- Use: the type system, color tokens, uppercase kicker labels for panel titles (`FINDINGS`, `PIPELINE MAP`), numbered finding indices, thin 1px borders, square-ish corners, pill tags for dialect names and pass names.
- **The workspace is one bordered grid**, like the reference's bento layout: headline bar, findings, pipeline map and inspector are cells sharing 1px rules, filling the viewport edge to edge. Each panel has a mono kicker label top-left and, where useful, a mono count/index top-right (`FINDINGS ········ 03`).
- **Findings render like the reference's cells:** mono index (`01`), sans title, gray one-line evidence, impact in mono.
- **The Pipeline Map uses the line-art diagram style:** op chips are thin bordered boxes with mono labels, connectors are thin orthogonal lines, approximate links are dashed, the selected lineage is drawn in cream with everything else dimmed.
- **The accent is reserved for exactly one thing at a time:** the current selection's primary state, i.e. the selected finding's `fix` button or the "verified" badge. Never decorative.
- Headline metric in the headline bar: large light sans with tabular numerals; its labels in small mono uppercase.
- Use sparingly: the token-field background and ASCII/glitch wordmark appear **only** on the start state. The ticker strip is not used in the workspace.
- No gradients, no drop shadows, no glassmorphism, no emoji, no icon soup. Icons only where text would be ambiguous.
- Motion: ≤150ms transitions on selection/highlight only. No looping animation in the workspace. Respect `prefers-reduced-motion`.
- Data colors: one heat scale for runtime share (single hue, varying intensity), one color for "changed", one for errors. Never rely on color alone: chips also show a numeric share on hover/selection, dashed lines for approximate links.

### 4.4 Starting tokens (estimated from a screenshot; replace with measured values)
```css
:root {
  --bg: #171412;        /* warm near-black page background */
  --bg-raised: #1d1a17; /* hover/selected cell */
  --rule: #2f2b27;      /* 1px grid lines and box borders */
  --fg: #ede7dd;        /* cream headlines */
  --body: #a8a199;      /* warm-gray body text */
  --dim: #6e6862;       /* indices, inactive labels, unselected chips */
  --accent: #5b16ff;    /* electric violet, solid fill, used rarely */
  --accent-fg: #ffffff;
  --error: #ff5c4d;
  /* heat scale for runtime share: cream at rising opacity on --bg (single hue) */
  --heat-1: rgba(237,231,221,0.08);
  --heat-5: rgba(237,231,221,0.55);
  --font-sans: "Geist", "Inter Tight", system-ui, sans-serif; /* light geometric sans; match the real one */
  --font-mono: "Geist Mono", "JetBrains Mono", ui-monospace, monospace;
  --radius: 0px;        /* reference uses square boxes */
}
```
Heat uses cream (not the accent) so the accent stays reserved for the selected action.
Map tokens into `tailwind.config` theme (`colors`, `fontFamily`, `borderRadius`) so components use Tailwind classes, never raw hex values. Support light theme only if the reference has one; otherwise ship dark only.

### 4.5 Typography scale (workspace)
- Headline metric: large light sans (weight 300), tabular numerals, cream.
- Section/start-state headlines: light sans, two-line split, centered.
- Body and evidence text: sans, ~15px, warm gray.
- Panel kicker labels, indices, pass names, dialect pills, diagram labels: mono, ~11px, uppercase, letter-spacing ~0.12em, `--dim` or `--body`.
- IR text: mono, 12–13px, line-height ~1.5, syntax highlighting with ≤4 colors (op names, SSA values, types, attributes).

---

## 5. Architecture

```
frontend (React + Vite + TS + Tailwind)
        │  REST + SSE (progress)
backend (FastAPI)
  ├── api/            routes, schemas (pydantic)
  ├── runner/         job queue (one worker process; compile jobs are CPU-heavy)
  ├── pipeline/       FX capture, torch-mlir import, per-pass MLIR execution, LLVM steps
  ├── provenance/     op extraction, location parsing, cross-stage link building
  ├── execute/        native build, ctypes harness, benchmark, correctness check
  ├── profile/        perf-based attribution, static cost model
  ├── detectors/      one module per detector
  ├── fixes/          fix definitions → pipeline patches
  └── store/          run storage on disk + SQLite index
toolchain: torch, torch-mlir (pinned), LLVM tools (clang, opt, llc, llvm-mca, llvm-objdump), perf (Linux)
```

- Compile jobs run in a **separate worker process** (not the FastAPI event loop). User model code runs in a subprocess with a timeout; it is untrusted.
- Progress streams to the UI via Server-Sent Events: `capture → import → passes (n/N) → build → benchmark → analyze → done`.
- Storage: `runs/<run_id>/` holds `run.json`, `snapshots/<idx>_<pass>.mlir`, `llvm/*.ll`, `asm/*.s`, `bin/model.so`, `perf/`, `reports/`. SQLite (`runs/index.db`) indexes runs, parents and status. Snapshots are stored in full (models are small); compress with zstd if size becomes a problem.

---

## 6. Compiler pipeline

### 6.1 Stages
| id | stage | produced by |
|---|---|---|
| `source` | PyTorch source + FX graph | `torch.export.export(model, inputs)`; FX nodes keep `stack_trace` → source lines |
| `torch` | Torch dialect | torch-mlir FX importer (`torch_mlir.fx.export_and_import`, raw output) |
| `linalg_tensor` | Linalg on tensors | torch-mlir backend lowering pipelines, then tensor-level optimizations |
| `linalg_buffer` | bufferized Linalg / memref | one-shot bufferization + deallocation |
| `loops` | SCF / Affine | Linalg → loops, loop transforms |
| `llvm_dialect` | LLVM dialect | conversion to LLVM |
| `llvm_ir` | LLVM IR | `mlir-translate --mlir-to-llvmir` (or Python equivalent), then `opt` |
| `asm` | assembly | `llc`/`clang` + `llvm-objdump` |

### 6.2 Pipeline config
The pipeline lives in `backend/pipelines/default.yaml`, never hard-coded. Each entry: `stage`, `pass` (fully anchored textual pipeline, e.g. `func.func(canonicalize)`), optional `options`. Example shape (pass names must be verified, see Rule 1):

```yaml
- stage: linalg_tensor
  passes:
    - "torch-backend-to-linalg-on-tensors-backend-pipeline"   # composite, see 6.3
    - "func.func(linalg-fuse-elementwise-ops)"
    - "canonicalize"
    - "cse"
- stage: linalg_buffer
  passes:
    - "one-shot-bufferize{bufferize-function-boundaries}"
    - "buffer-deallocation-pipeline"
    - "canonicalize"
- stage: loops
  passes:
    - "func.func(convert-linalg-to-affine-loops)"
    - "func.func(affine-loop-fusion)"
    - "lower-affine"
    - "convert-scf-to-cf"
- stage: llvm_dialect
  passes:
    - "expand-strided-metadata"
    - "finalize-memref-to-llvm"
    - "convert-to-llvm"
    - "reconcile-unrealized-casts"
```

### 6.3 Per-pass snapshots
- Run passes **one at a time** through the Python bindings: for each entry, `PassManager.parse("builtin.module(<pass>)").run(module.operation)`, then snapshot `module.operation.get_asm(enable_debug_info=True)`.
- Record per pass: wall time, `changed` (text hash differs), op counts by op name, total op count.
- Composite pipelines from torch-mlir: try to expand them into their component passes (use the pipeline's textual expansion if the bindings expose it). If they cannot be split, snapshot once after the composite and mark it `composite: true` in the UI.
- Mark the `func` that will be called from the harness with `llvm.emit_c_interface` so a C-callable wrapper is generated.

### 6.4 LLVM and below
- Translate LLVM dialect to LLVM IR. Run `opt -passes='default<O3>'` with `-print-changed` captured per pass (same snapshot format as MLIR passes, stage `llvm_ir`).
- Collect `-pass-remarks-missed` output (at least for `loop-vectorize`) into `reports/remarks.txt`.
- Build `model.so` with `clang -O3 -g -shared -fPIC -march=native`. Emit assembly with `llvm-objdump -d --no-show-raw-insn`.
- Version note: the MLIR bundled in torch-mlir and the system LLVM tools must be compatible at the textual LLVM IR level. The spike (M0) checks this; document the working combination in `docs/toolchain.md`.

---

## 7. Provenance (op lineage)

- Every op in every snapshot gets a stable id: `<stage>:<snapshot_idx>:<op_index_in_walk_order>`, plus name, result types, operand SSA names, and its location.
- Root identity: FX node name + source file:line from `stack_trace`. The FX importer attaches locations to Torch dialect ops; keep them through every pass by always printing with debug info.
- **Links:** between consecutive snapshots, link ops that share a location. `loc(fused[...])` produces links to every contributing parent (this is how fusion shows up). `callsite` and `name` locs are unwrapped.
- **Lost locations:** when an op has `loc(unknown)` or a location not seen before, attempt a heuristic match (same parent region, dataflow neighbors with known lineage). Heuristic links are stored with `approximate: true` and drawn dashed. Never present a heuristic link as exact.
- LLVM IR: MLIR locations become debug info (`!dbg`). Map instructions → MLIR lines in the `llvm_dialect` snapshot file → op ids. Assembly maps through DWARF line info the same way.
- Expose a coverage metric per stage: `% ops with exact lineage`. Show it in the pipeline map row label. This honesty is a feature.

---

## 8. Measurement and analysis

### 8.1 Execution and correctness
- Harness: load `model.so` with `ctypes`, build memref descriptors from NumPy arrays (rank-specific structs: allocated ptr, aligned ptr, offset, sizes, strides), call the `_mlir_ciface_` wrapper.
- Correctness: compare outputs with PyTorch eager using `allclose` (rtol/atol configurable, defaults 1e-4/1e-5 for fp32). Failure blocks all speed claims for that run.
- Benchmark: warmup 10, then N=100 iterations (configurable), report median and p10/p90. Same protocol for eager, with `torch.set_num_threads(T)` matching the compiled code's thread count (default 1). Pin to one core if `taskset` is available.

### 8.2 Runtime attribution
- Primary: `perf record` over the benchmark loop, then resolve samples to source lines via debug info → `llvm_dialect` snapshot lines → op ids → lineage → source line. Store per-op and per-source-line sample shares.
- If `perf` is unavailable (non-Linux, permissions), fall back to static cost estimates (8.3) and label every share as `est`. Show a one-line notice explaining why.

### 8.3 Static cost model (per stage, per op)
- FLOPs from Linalg iteration domains × arithmetic ops in the region body.
- Bytes moved: sum of operand and result tensor/memref sizes per op.
- Arithmetic intensity = FLOPs / bytes.
- Allocations: count and total bytes of `memref.alloc`; peak live bytes from a simple liveness walk.
- Vectorization: presence of `vector` dialect ops at `loops`; vector types (`<N x float>`) in LLVM IR; SIMD instructions in asm for hot functions.

---

## 9. Detectors and fixes (v1 ships exactly three)

Each detector returns findings with: `id`, `title`, `op_ids` (all stages), `evidence` (facts + raw tool output excerpt), `impact_estimate`, `fix_ids`. A detector that is not certain does not emit a finding.

**D1 — Unfused elementwise chain**
- Rule: at the end of `linalg_tensor`, a `linalg.generic` with all-parallel iterators whose result has a single use by another Linalg op with the same iteration domain, and the two were not fused.
- Evidence: intermediate tensor shape and bytes; the passes that ran; lineage of both ops.
- Impact estimate: intermediate bytes × 2 (write + read) ÷ measured memory bandwidth of a calibration kernel (run once per machine, cached).
- Fix: enable/reorder elementwise fusion, or apply a transform script fusing the producer into the consumer.

**D2 — Unnecessary copy / allocation**
- Rule: at `linalg_buffer`, `memref.copy` ops or `memref.alloc` on the hot path that bufferization's analysis reports as caused by an in-place conflict.
- Evidence: re-run one-shot bufferization in analysis-only mode with conflict printing (verify exact option names) and attach the conflict report for the op.
- Impact estimate: copy bytes ÷ measured bandwidth.
- Fix: enable empty-tensor elimination, function-boundary bufferization, or the specific option the conflict report points to.

**D3 — Hot loop not vectorized**
- Rule: an op with ≥10% runtime share whose innermost loop has no vector types in LLVM IR and no SIMD instructions in asm.
- Evidence: LLVM missed-optimization remark for that loop (if any), the loop nest at `loops`, the asm excerpt.
- Impact estimate: conservative, from vector width × fraction of time in that loop, capped and labeled `est`.
- Fix: tile + vectorize via a transform-dialect script (tile innermost dims to the target vector width, then vectorize), or the equivalent pass options.

**Fixes**
- A fix is a pipeline patch: insert/replace/remove entries in the pipeline config, change pass options, or attach a transform script. Stored as data, diffable, shown in Compare.
- Applying a fix = new child run with the patched pipeline. When it completes, the parent's finding shows `verified: <measured ratio>` next to `est`. If the fix made it slower, say so plainly.

---

## 10. Data model (pydantic, mirrored in TS types)

```
Run        { id, parent_id?, created_at, source_code, inputs_spec, pipeline (config), fix_id?,
             status, error?, metrics: Metrics, stages: Stage[] }
Stage      { id, name, snapshot_ids[], exact_lineage_pct }
Snapshot   { idx, stage, pass, composite, changed, time_ms, op_counts{}, path }
Op         { id, snapshot_idx, name, loc, source_line?, result_types[], cost? }
Link       { from_op, to_op, approximate }
Metrics    { runtime_ms_median, p10, p90, eager_ms_median, threads, correct, max_abs_err,
             peak_bytes, alloc_count, per_line_share{}, per_op_share{}, attribution: "perf"|"est" }
Finding    { id, detector, title, op_ids[], evidence, impact_est, impact_verified?, fix_ids[] }
Fix        { id, label, pipeline_patch, transform_script? }
```
Ops and links for a run are served lazily per stage (they can be large).

## 11. API

| method | path | purpose |
|---|---|---|
| POST | `/api/runs` | create run `{source, pipeline?, parent_id?, fix_id?}` → `{run_id}` |
| GET | `/api/runs` | list runs (tree) |
| GET | `/api/runs/{id}` | run summary, metrics, findings |
| GET | `/api/runs/{id}/events` | SSE progress |
| GET | `/api/runs/{id}/stages/{stage}` | ops + links for a stage |
| GET | `/api/runs/{id}/snapshots/{idx}` | raw snapshot text |
| GET | `/api/runs/{id}/diff?a=&b=` | pass diff or run-vs-run stage diff |
| POST | `/api/runs/{id}/findings/{fid}/apply` | apply fix → child run id |
| GET | `/api/runs/{id}/reproducer` | zip |
| GET | `/api/templates`, `/api/templates/{name}` | model templates |
| GET | `/api/toolchain` | versions + capability check (perf available, vector ISA, etc.) |

---

## 11b. Dependencies (allowed)
- Backend: `fastapi`, `uvicorn`, `pydantic`, `torch` (CPU), `torch-mlir` (pinned nightly), `numpy`, `pyyaml`, `sse-starlette`, `zstandard` (optional), `pytest`.
- Frontend: `react`, `react-dom`, `vite`, `typescript`, `tailwindcss`, `@codemirror/*` (editor, IR view, merge/diff view), `zustand` (state), `@tanstack/react-query` (server data). Pipeline Map is custom SVG/DOM — no graph library.

---

## 12. Milestones (build in order)

**M0 — Toolchain spike (backend only, no UI)**
Script `scripts/spike.py`: take the `matmul_relu` template, go all the way to `model.so`, run it, check correctness vs eager, time both, print per-pass changed/op counts.
✅ Done when: one command prints correct output, two runtimes, and a pass table for every stage through asm. Toolchain versions recorded in `docs/toolchain.md`.

**M1 — Run pipeline + storage + API**
Pipeline config, per-pass snapshots, run storage, SSE progress, `/api/runs*`, `/api/toolchain`.
✅ Done when: POST a template → run completes → all snapshots retrievable; pytest covers 3 templates.

**M2 — Workspace UI skeleton + design tokens**
Measured tokens (Section 4.1), start state, headline bar, Inspector with raw IR + pass diff.
✅ Done when: compile from the browser, see headline metrics, browse any pass snapshot and its diff.

**M3 — Provenance + Pipeline Map**
Op extraction, location links, approximate links, coverage metric, click-to-trace lineage across all stages.
✅ Done when: clicking `L14 softmax` highlights its ops in every stage including asm; coverage % shown per stage.

**M4 — Attribution + cost model**
perf attribution with fallback; heat coloring; hottest-line headline.
✅ Done when: hot op shares sum to ≈100% (±5%) on Linux; fallback path labeled `est`.

**M5 — Detectors D1–D3 + findings list**
✅ Done when: each detector has a positive and a negative test model; zero findings on a model with nothing to fix.

**M6 — Fixes + run history + Compare**
✅ Done when: on the `attention_block` template, applying the top finding produces a child run that is measurably faster and the finding shows `verified`.

**M7 — Advanced drawer + reproducer export + polish**
Pipeline editor, transform script editor, settings, keyboard shortcuts, empty/error states.

---

## 13. Templates (in `backend/templates/`)
`matmul_relu.py`, `mlp.py` (3 layers), `softmax.py`, `layernorm.py`, `attention_block.py` (single-head, small dims). Each defines `model` and `inputs`. These double as the test suite and the demo.

## 14. Testing
- Backend: pytest for pipeline (every template compiles and is correct), provenance (known op has lineage to asm), each detector (positive/negative), fix application (child run differs only by the patch).
- Frontend: vitest for parsers/state; one Playwright smoke test (compile template → findings visible → apply fix → compare opens).

## 15. Demo script (target: under 2 minutes)
1. Load `attention_block` → compile.
2. Headline: slower than eager, most time in softmax.
3. Click finding `01` → Pipeline Map lights softmax across all stages; evidence shows the unfused chain and intermediate buffer.
4. Press `f` → child run compiles.
5. Compare: measured speedup, IR diff, pipeline patch. Finding shows `verified`.

## 16. Out of scope for v1 (do not build)
GPU targets · IREE/StableHLO backends (possible v2 stage provider) · training/backward graphs · Hugging Face model loading · user accounts · cloud deployment · AI-generated explanations · autotuning search · multi-threaded codegen.
