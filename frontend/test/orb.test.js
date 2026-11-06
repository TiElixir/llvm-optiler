import test from "node:test";
import assert from "node:assert/strict";
import { orbStateForPhase } from "../js/orb.js";
import { STATE_TO_MODE, resolvePreset } from "../vendor/thinking-orbs/presets.js";
import { MODE_DRAWS } from "../vendor/thinking-orbs/engine/registry.js";

test("every pipeline phase maps to one of the six shipped orb states", () => {
  const phases = ["queued", "capture", "import", "passes", "build", "provenance", "benchmark", "analyze", "done", "error"];
  for (const p of phases) assert.ok(orbStateForPhase(p) in STATE_TO_MODE, p);
  assert.notEqual(orbStateForPhase("passes"), orbStateForPhase("build"));
  assert.equal(orbStateForPhase("something-new"), "working");
});

test("vendored engine resolves both sizes of every state to a drawable mode", () => {
  for (const state of Object.keys(STATE_TO_MODE)) {
    for (const size of [20, 64]) {
      const r = resolvePreset(state, size);
      assert.equal(typeof MODE_DRAWS[r.mode], "function", `${state}@${size}`);
      assert.ok(r.speed > 0);
    }
  }
});

test("each state draws without throwing on a recording canvas context", () => {
  const calls = [];
  const ctx = new Proxy({}, { get: (_t, k) => (k in ctx2 ? ctx2[k] : (...a) => { calls.push(k); }), set: () => true });
  const ctx2 = {};
  for (const state of Object.keys(STATE_TO_MODE)) {
    for (const size of [20, 64]) {
      const r = resolvePreset(state, size);
      MODE_DRAWS[r.mode](ctx, size, 1.234, true, r.opts);
    }
  }
  assert.ok(calls.length > 100); // dots were drawn
});
