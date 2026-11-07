import test from "node:test";
import assert from "node:assert/strict";
import { attachTransform, passRows, readSettings } from "../js/drawer.js";

const YAML = `- stage: torch
  passes:
  - inline
- stage: linalg_tensor
  passes:
  - torch-backend-to-linalg-on-tensors-backend-pipeline{allow-non-finites=true}
  - cse
- stage: linalg_buffer
  passes:
  - one-shot-bufferize
`;

test("attachTransform adds the preload and interpreter at the end of linalg_tensor, matching indentation", () => {
  const out = attachTransform(YAML).split("\n");
  const i = out.findIndex((l) => l.includes("transform-preload-library"));
  assert.equal(out[i - 1], "  - cse");
  assert.equal(out[i], '  - "transform-preload-library{transform-library-paths=${RUN_DIR}/transforms/user.mlir}"');
  assert.equal(out[i + 1], "  - transform-interpreter");
  assert.equal(out[i + 2], "- stage: linalg_buffer");
});

test("attachTransform is idempotent and leaves pipelines without the stage alone", () => {
  const once = attachTransform(YAML);
  assert.equal(attachTransform(once), once);
  const no = "- stage: torch\n  passes:\n  - inline\n";
  assert.equal(attachTransform(no), no);
});

test("attachTransform works when linalg_tensor is the last stage", () => {
  const out = attachTransform("- stage: linalg_tensor\n  passes:\n  - cse\n");
  assert.ok(out.trimEnd().endsWith("- transform-interpreter"));
});

test("readSettings validates integers and bounds", () => {
  assert.deepEqual(readSettings({ iters: "100", warmup: "10", threads: "1" }), { ok: true, values: { iters: 100, warmup: 10, threads: 1 } });
  assert.equal(readSettings({ iters: "0", warmup: "1", threads: "1" }).ok, false);
  assert.equal(readSettings({ iters: "5", warmup: "-1", threads: "1" }).ok, false);
  assert.equal(readSettings({ iters: "5", warmup: "1", threads: "1.5" }).ok, false);
  assert.match(readSettings({ iters: "abc", warmup: "1", threads: "1" }).error, /iterations/);
});

test("passRows flags the slowest passes", () => {
  const snaps = Array.from({ length: 10 }, (_, i) => ({ idx: i, stage: "loops", pass: `p${i}`, changed: false, time_ms: i }));
  const rows = passRows(snaps);
  assert.deepEqual(rows.filter((r) => r.slow).map((r) => r.idx), [5, 6, 7, 8, 9]);
  assert.equal(rows[0].slow, false);
});
