import test from "node:test";
import assert from "node:assert/strict";
import { evidenceLine, evidenceRows, findingLines, impactText, indexLabel, stepIndex } from "../js/findings.js";

const f = (o = {}) => ({ id: "x", impact_est: { ratio: 2.145, kind: "est" }, impact_verified: null,
  evidence: { why: "the loop stays scalar", lines: [20, 23], hot_asm: ["a", "b"], runtime_share: { 20: 0.43 }, bytes: 4096, nothing: null }, ...o });

test("impact reads est until a fix is verified, then the measured outcome", () => {
  assert.equal(impactText(f()), "est 2.15×");
  assert.equal(impactText(f({ impact_verified: { ratio: 13.04, kind: "verified" } })), "verified 13.0×");
  assert.equal(impactText(f({ impact_verified: { ratio: 0.5, kind: "slower" } })), "slower 2.00×");
  assert.equal(impactText(f({ impact_verified: { ratio: 1, kind: "broken" } })), "broke correctness");
  assert.equal(impactText(f({ impact_verified: { ratio: 1.01, kind: "none" } })), "no measurable change");
});

test("index labels, evidence line, lines to trace", () => {
  assert.equal(indexLabel(0), "01"); assert.equal(indexLabel(11), "12");
  assert.equal(evidenceLine(f()), "the loop stays scalar");
  assert.equal(evidenceLine(f({ evidence: { a: 1, b: 2 } })), "a, b");
  assert.deepEqual(findingLines(f()), [20, 23]); assert.deepEqual(findingLines(f({ evidence: {} })), []);
});

test("evidence rows classify values and skip nulls", () => {
  const rows = evidenceRows(f());
  const by = Object.fromEntries(rows.map((r) => [r[0], r]));
  assert.equal(by.nothing, undefined);
  assert.deepEqual(by.hot_asm.slice(1), ["code", ["a", "b"]]);
  assert.equal(by.runtime_share[1], "list"); assert.deepEqual(by.runtime_share[2], ["20: 0.43"]);
  assert.deepEqual(by.bytes, ["bytes", "text", "4096"]);
});

test("stepIndex clamps and handles empty lists", () => {
  assert.equal(stepIndex(-1, 1, 3), 0); assert.equal(stepIndex(0, -1, 3), 0); assert.equal(stepIndex(2, 1, 3), 2); assert.equal(stepIndex(0, 1, 0), -1);
});
