import test from "node:test";
import assert from "node:assert/strict";
import { esc, fmtBytes, fmtMs, fmtPct, fmtRatio, pad2 } from "../js/format.js";

test("fmtMs picks sensible precision", () => {
  assert.equal(fmtMs(0.5184), "0.518");
  assert.equal(fmtMs(3.2147), "3.21");
  assert.equal(fmtMs(28.56), "28.6");
  assert.equal(fmtMs(412.2), "412");
  assert.equal(fmtMs(null), "–");
  assert.equal(fmtMs(NaN), "–");
});

test("ratios, bytes, percentages, indices, escaping", () => {
  assert.equal(fmtRatio(1.84), "1.84");
  assert.equal(fmtRatio(13.04), "13.0");
  assert.equal(fmtBytes(262208), "256.1 KiB");
  assert.equal(fmtBytes(12), "12 B");
  assert.equal(fmtPct(0.64), "64%");
  assert.equal(pad2(3), "03");
  assert.equal(esc('<a href="x">&</a>'), "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
});
