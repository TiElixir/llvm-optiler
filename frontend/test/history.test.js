import test from "node:test";
import assert from "node:assert/strict";
import { compareRows, runLabel, treeOf } from "../js/history.js";

const run = (id, seq, parent_id = null, o = {}) => ({ id, seq, parent_id, label: "", status: "done", runtime_ms_median: 1.5, correct: true, ...o });

test("treeOf returns the whole tree of a run, oldest first, and nothing from other trees", () => {
  const runs = [run("a", 1), run("b", 2, "a"), run("c", 3, "b"), run("x", 4), run("y", 5, "x")];
  assert.deepEqual(treeOf(runs, "c").map((r) => r.id), ["a", "b", "c"]);
  assert.deepEqual(treeOf(runs, "a").map((r) => r.id), ["a", "b", "c"]);
  assert.deepEqual(treeOf(runs, "y").map((r) => r.id), ["x", "y"]);
  assert.deepEqual(treeOf(runs, "zzz"), []);
});

test("runLabel shows number, fix label and runtime; unfinished and wrong runs say so", () => {
  assert.deepEqual(runLabel(run("a", 2, "a", { label: "interchange matmul loops", runtime_ms_median: 0.5184 })), { index: "#2", name: "interchange matmul loops", time: "0.518 ms" });
  assert.equal(runLabel(run("a", 1, null, { status: "running" })).time, "running");
  assert.equal(runLabel(run("a", 1, null, { correct: false })).time, "wrong output");
});

const metrics = (o = {}) => ({ correct: true, runtime_ms_median: 6.7, runtime_ms_p10: 6.4, runtime_ms_p90: 7.0, ratio_vs_eager: 28, peak_bytes: 246784, alloc_count: 9, ...o });
const cmp = (a, b) => ({ a_metrics: metrics(a), b_metrics: metrics(b), stages: [{ stage: "loops", a_ops: 323, b_ops: 280 }, { stage: "asm", a_ops: 100, b_ops: 100 }] });

test("compareRows: speed rows, stage op counts, and no deltas for unchanged counts", () => {
  const rows = compareRows(cmp({}, { runtime_ms_median: 0.52, runtime_ms_p10: 0.5, runtime_ms_p90: 0.53, ratio_vs_eager: 2.2 }));
  const by = Object.fromEntries(rows.map((r) => [r[0], r]));
  assert.equal(by["runtime (ms)"][3], "12.9× faster");
  assert.equal(by["ops · loops"][3], "−13%"); assert.equal(by["ops · asm"][3], "");
  assert.equal(by["peak memory"][1], "241.0 KiB");
});

test("compareRows hides speed claims when either run is incorrect", () => {
  const rows = compareRows(cmp({}, { correct: false }));
  assert.equal(rows.some((r) => r[0] === "runtime (ms)" || r[0] === "vs eager"), false);
  assert.deepEqual(rows[0], ["correct", "true", "false", ""]);
});

test("a slower child reads slower, not a negative speedup", () => {
  const by = Object.fromEntries(compareRows(cmp({}, { runtime_ms_median: 13.4, runtime_ms_p10: 13, runtime_ms_p90: 14 })).map((r) => [r[0], r]));
  assert.equal(by["runtime (ms)"][3], "2.00× slower");
});
