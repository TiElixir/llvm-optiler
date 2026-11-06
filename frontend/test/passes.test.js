import test from "node:test";
import assert from "node:assert/strict";
import { countUnit, describeDelta, opDelta, stageSnapshots } from "../js/passes.js";

const snap = (idx, counts, stage = "linalg_tensor") => ({
  idx, stage, op_counts: counts, total_ops: Object.values(counts).reduce((a, b) => a + b, 0),
});

test("opDelta lists changed op names, largest change first", () => {
  const d = opDelta(snap(1, { "linalg.generic": 6, "arith.constant": 3 }),
                    snap(2, { "linalg.generic": 3, "arith.constant": 3, "tensor.empty": 1 }));
  assert.deepEqual(d.total, [9, 7]);
  assert.deepEqual(d.changed, [["linalg.generic", 6, 3], ["tensor.empty", 0, 1]]);
});

test("opDelta handles a missing previous snapshot", () => {
  assert.deepEqual(opDelta(undefined, snap(0, { "torch.aten.mm": 1 })).changed, [["torch.aten.mm", 0, 1]]);
});

test("describeDelta uses ops for MLIR and lines for LLVM", () => {
  const d = opDelta(snap(1, { "linalg.generic": 6 }), snap(2, { "linalg.generic": 3 }));
  assert.equal(describeDelta(d, "linalg_tensor"), "6 → 3 ops: linalg.generic 6 → 3");
  assert.equal(describeDelta(d, "llvm_ir"), "6 → 3 lines");
  assert.equal(countUnit("asm"), "lines");
  assert.equal(describeDelta(opDelta(snap(1, { "a.b": 2 }), snap(2, { "a.b": 2 })), "loops"), "2 → 2 ops, no op-count change");
  assert.ok(describeDelta(opDelta(snap(1, {}), snap(2, { a: 1, b: 1, c: 1, d: 1, e: 1, f: 1 })), "loops", 2).includes("+4 more"));
});

test("stageSnapshots pairs each snapshot with its predecessor in the run", () => {
  const rows = stageSnapshots([snap(0, {}), snap(1, {}), snap(2, {})], [1, 2]);
  assert.deepEqual(rows.map((r) => [r.snap.idx, r.prev?.idx]), [[1, 0], [2, 1]]);
});
