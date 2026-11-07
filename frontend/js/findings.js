// Findings: formatting helpers (pure, tested).
import { fmtRatio, pad2 } from "./format.js";

/** `est 2.1×` or, once a fix has been applied and measured, `verified 13.0×`. */
export function impactText(f) {
  const v = f.impact_verified;
  if (v) {
    if (v.kind === "verified") return `verified ${fmtRatio(v.ratio)}×`;
    if (v.kind === "slower") return `slower ${fmtRatio(1 / v.ratio)}×`;
    if (v.kind === "broken") return "broke correctness";
    return "no measurable change";
  }
  return `est ${fmtRatio(f.impact_est.ratio)}×`;
}

export const indexLabel = (i) => pad2(i + 1);

/** One-line evidence under the title. */
export function evidenceLine(f) {
  const e = f.evidence ?? {};
  if (e.why) return e.why;
  return Object.keys(e).slice(0, 3).join(", ");
}

/** Source lines a finding is about (to trace in the map). */
export const findingLines = (f) => (f.evidence?.lines ?? []).filter(Number.isInteger);

/** Evidence as display rows: [key, kind, value] with kind in text | list | code. */
export function evidenceRows(f) {
  const rows = [];
  for (const [k, v] of Object.entries(f.evidence ?? {})) {
    if (v == null) continue;
    if (Array.isArray(v) && v.every((x) => typeof x === "string")) rows.push([k, k.includes("asm") ? "code" : "list", v]);
    else if (Array.isArray(v)) rows.push([k, "code", v.map((x) => JSON.stringify(x))]);
    else if (typeof v === "object") rows.push([k, "list", Object.entries(v).map(([a, b]) => `${a}: ${b}`)]);
    else rows.push([k, "text", String(v)]);
  }
  return rows;
}

/** Next/previous index for j/k, clamped. */
export const stepIndex = (cur, dir, n) => (n === 0 ? -1 : cur < 0 ? 0 : Math.min(n - 1, Math.max(0, cur + dir)));
