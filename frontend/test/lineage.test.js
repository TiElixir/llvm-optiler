import test from "node:test";
import assert from "node:assert/strict";
import { connectorPath, coverageText, groupHit, groupOps, groupShare, heatBucket, highlightTextLines, toggleSelection } from "../js/lineage.js";

const op = (id, name, lines, extra = {}) => ({ id, name, source_lines: lines, displayable: true, approximate: false, text_line: null, ...extra });
const stage = { stage: "linalg_tensor", ops: [
  op("a:1:0", "linalg.matmul", [7], { text_line: 12 }),
  op("a:1:1", "linalg.generic", [8], { text_line: 15 }),
  op("a:1:2", "linalg.generic", [8], { text_line: 20, approximate: true }),
  op("a:1:3", "linalg.fill", [7, 8], { text_line: 11 }),
  op("a:1:4", "x.orphan", []),
  op("a:1:5", "arith.constant", [7], { displayable: false }),
] };

test("groupOps groups by line set, aggregates chips, orders by line, unattributed last", () => {
  const g = groupOps(stage);
  assert.deepEqual(g.map((x) => x.label), ["L7", "L7+L8", "L8", "no line"]);
  const l8 = g.find((x) => x.key === "8");
  assert.deepEqual(l8.chips, [{ name: "linalg.generic", count: 2, approx: true, opIds: ["a:1:1", "a:1:2"] }]);
  assert.equal(l8.approximate, true);
  assert.equal(g.find((x) => x.key === "7").chips.length, 1); // constants are not displayable
});

test("groupOps caps chips per group and keeps the rest in hidden", () => {
  const ops = Array.from({ length: 20 }, (_, i) => op(`s:0:${i}`, `op.n${i}`, [3]));
  const [g] = groupOps({ ops });
  assert.equal(g.chips.length, 14); assert.equal(g.hidden.length, 6); assert.equal(g.total, 20);
});

test("groupHit: a fused group is hit by any of its lines", () => {
  const [l7, fused] = groupOps(stage);
  assert.equal(groupHit(l7, [7]), true);
  assert.equal(groupHit(fused, [8]), true);
  assert.equal(groupHit(l7, [8]), false);
  assert.equal(groupHit(l7, []), false);
});

test("highlightTextLines returns sorted snapshot lines of the selected origin", () => {
  assert.deepEqual(highlightTextLines(stage, [8]), [11, 15, 20]);
  assert.deepEqual(highlightTextLines(stage, []), []);
});

test("toggleSelection selects, re-click clears, empty lines never select", () => {
  assert.deepEqual(toggleSelection([], [8]), [8]);
  assert.deepEqual(toggleSelection([8], [8]), []);
  assert.deepEqual(toggleSelection([8], [7, 8]), [7, 8]);
  assert.deepEqual(toggleSelection([8], []), []);
});

test("connectorPath is orthogonal and straight when aligned", () => {
  assert.equal(connectorPath({ x: 0, y: 0, width: 20, height: 10 }, { x: 0, y: 50, width: 20, height: 10 }), "M10 10V50");
  assert.equal(connectorPath({ x: 0, y: 0, width: 20, height: 10 }, { x: 100, y: 50, width: 20, height: 10 }), "M10 10V30H110V50");
});

test("coverageText", () => {
  assert.equal(coverageText(0.456), "46% exact lineage"); assert.equal(coverageText(null), "");
});

test("groupShare sums a group's lines; heatBucket thresholds", () => {
  const [l7, fused] = groupOps(stage);
  assert.equal(groupShare(l7, { 7: 0.25, 8: 0.5 }), 0.25);
  assert.equal(groupShare(fused, { 7: 0.25, 8: 0.5 }), 0.75);
  assert.equal(groupShare(l7, {}), null);
  assert.equal(groupShare(l7, null), null);
  assert.deepEqual([null, 0, 0.01, 0.05, 0.15, 0.3, 0.9].map(heatBucket), [0, 0, 1, 2, 3, 4, 5]);
});
