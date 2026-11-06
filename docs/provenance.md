# Provenance (op lineage)

What the Pipeline Map shows, and how far to trust it.

**Identity.** The unit of lineage is a *line of the user's source* (`source/model.py`). The FX importer
records `loc("source/model.py":LINE:0)`; `nn.Linear` and friends arrive as `callsite(<torch file> at
<source/model.py:LINE>)`, and the caller frame is the user's line. Several FX nodes can share a line (for
example `torch.relu(y + b)`), and one line cannot be split further: that is the finest identity the
locations carry. `fused[...]` locations (from fusion) stand for every contributing line, and an op with
several lines is shown once, under `L7+L8`.

**Per stage.** At the end of each stage the module is walked in print order. The same module is printed
once more by `snapshot-op-locations{print-debuginfo=true}` on a clone, so each op also knows its *exact*
line in the stored snapshot text (constants elided without changing line numbers). That is what the
inspector highlights.

**Below the LLVM dialect.** The anchor file `anchor/llvm_dialect.mlir` is the text whose line numbers
replace the op locations before translation. `!DILocation(line: N)` in the LLVM IR, `DebugLoc` in
optimization remarks and the `; …/llvm_dialect.mlir:N` markers from `llvm-objdump -l` all name an anchor
line, hence a dialect op, hence its source lines. Instructions with no `!dbg` (frame setup, vector
runtime checks, loop remainders) and assembly before the first marker have no source line and count
against coverage; they are listed under "no line".

**Exact vs approximate.** An op whose location carries no user line gets lines from its operand
producers, then its users, then the nearest preceding sibling or its parent. Those ops are marked
`approximate` (drawn dashed, labeled `~`) and do not count as exact. They are never presented as
compiler truth.

**Coverage.** `exact_lineage_pct` per stage = displayable ops with a recorded source line / displayable
ops. Displayable excludes containers (`func.func`, `module`, `memref.global`), constants, terminators and
`linalg` payload ops (which stay in the table so highlighting still works). Measured on `attention_block`:
torch 100%, linalg 100%, buffers 100%, loops 93%, LLVM dialect 98%, LLVM IR 86%, assembly 72%.

**Links.** `links` between consecutive stages are derived from shared source lines (LLVM IR → dialect
through the exact anchor line). Each op carries at most four links; a dense group's full relation is in
`source_lines`, which is what the UI uses.
