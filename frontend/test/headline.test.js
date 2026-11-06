import test from "node:test";
import assert from "node:assert/strict";
import { headlineModel, pluralThreads } from "../js/headline.js";

const metrics = (o = {}) => ({ correct: true, runtime_ms_median: 3.2147, ratio_vs_eager: 1.84, threads: 1,
  max_abs_err: 1e-6, attribution: null, hottest_line: null, ...o });
const run = (o = {}) => ({ status: "done", phase: "done", error: null, error_stage: null, metrics: metrics(), ...o });

test("queued/running runs show progress and no numbers", () => {
  const m = headlineModel(run({ status: "running", phase: "passes" }), { phase: "passes", n: 3, total: 12, name: "canonicalize" });
  assert.deepEqual(m, { kind: "progress", phase: "passes", step: "3/12", name: "canonicalize" });
  assert.equal(headlineModel(run({ status: "queued", phase: "queued" }), null).step, null);
});

test("a compile error reports the stage and the first three lines", () => {
  const m = headlineModel(run({ status: "error", error_stage: "loops", error: "a\n\nb\nc\nd" }), null);
  assert.equal(m.kind, "failed"); assert.equal(m.title, "compile failed at loops"); assert.deepEqual(m.lines, ["a", "b", "c"]);
});

test("a correctness mismatch hides every speed number", () => {
  const m = headlineModel(run({ metrics: metrics({ correct: false, max_abs_err: 0.5 }) }), null);
  assert.equal(m.kind, "mismatch"); assert.equal(m.text, "output mismatch at native build (max abs err 0.500)");
  assert.equal("runtime" in m, false);
});

test("ratio > 1 reads slower, < 1 reads faster, thread count is stated", () => {
  const slow = headlineModel(run(), null);
  assert.equal(slow.vsEager.text, "1.84× slower than eager"); assert.equal(slow.vsEager.threads, "1 thread"); assert.equal(slow.runtime, "3.21");
  const fast = headlineModel(run({ metrics: metrics({ ratio_vs_eager: 0.5, threads: 4 }) }), null);
  assert.equal(fast.vsEager.text, "2.00× faster than eager"); assert.equal(fast.vsEager.threads, "4 threads");
  assert.equal(pluralThreads(1), "1 thread"); assert.equal(pluralThreads(2), "2 threads");
});

test("no speed claim until correctness is confirmed", () => {
  const m = headlineModel(run({ metrics: metrics({ correct: null }) }), null);
  assert.equal(m.vsEager, null); assert.equal(m.unverified, true);
});

test("hottest line only when the backend measured one, labeled est or measured", () => {
  assert.equal(headlineModel(run(), null).hottest, null);
  const est = headlineModel(run({ metrics: metrics({ hottest_line: 14, hottest_line_share: 0.64, hottest_line_code: "attn = softmax(s)", attribution: "est" }) }), null);
  assert.deepEqual(est.hottest, { share: "64%", line: 14, code: "attn = softmax(s)", attribution: "est" });
  const probe = headlineModel(run({ metrics: metrics({ hottest_line: 7, hottest_line_share: 0.9, hottest_line_code: "y = x @ w", attribution: "probe" }) }), null);
  assert.equal(probe.hottest.attribution, "measured");
});
