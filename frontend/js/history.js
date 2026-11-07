// Run history and Compare: pure helpers (tested).
import { fmtBytes, fmtMs, fmtRatio } from "./format.js";

/** Runs of the tree the given run belongs to (root ancestor and all its descendants), oldest first. */
export function treeOf(runs, id) {
  const byId = new Map(runs.map((r) => [r.id, r]));
  let root = byId.get(id);
  while (root?.parent_id && byId.has(root.parent_id)) root = byId.get(root.parent_id);
  if (!root) return [];
  const members = new Set([root.id]);
  for (let grew = true; grew;) {
    grew = false;
    for (const r of runs) if (r.parent_id && members.has(r.parent_id) && !members.has(r.id)) { members.add(r.id); grew = true; }
  }
  return runs.filter((r) => members.has(r.id)).sort((a, b) => a.seq - b.seq);
}

/** `#2 interchange matmul loops 0.52 ms` */
export function runLabel(r) {
  const name = r.label || "run";
  const t = r.status === "done" ? (r.correct === false ? "wrong output" : `${fmtMs(r.runtime_ms_median)} ms`) : r.status;
  return { index: `#${r.seq}`, name, time: t };
}

const pct = (a, b) => (a && b ? `${b / a >= 1 ? "+" : "−"}${Math.abs((b / a - 1) * 100).toFixed(0)}%` : "");

/** Rows for the compare table: [label, a, b, delta]. Speed rows are hidden unless both runs are correct. */
export function compareRows(c) {
  const A = c.a_metrics, B = c.b_metrics;
  const ok = A.correct === true && B.correct === true;
  const rows = [["correct", String(A.correct), String(B.correct), ""]];
  if (ok) {
    rows.push(["runtime (ms)", `${fmtMs(A.runtime_ms_median)} (${fmtMs(A.runtime_ms_p10)}–${fmtMs(A.runtime_ms_p90)})`,
      `${fmtMs(B.runtime_ms_median)} (${fmtMs(B.runtime_ms_p10)}–${fmtMs(B.runtime_ms_p90)})`,
      `${fmtRatio(A.runtime_ms_median / B.runtime_ms_median)}× ${A.runtime_ms_median >= B.runtime_ms_median ? "faster" : "slower"}`.replace(/^(\S+)× (faster|slower)$/, (m, r, d) => d === "slower" ? `${fmtRatio(B.runtime_ms_median / A.runtime_ms_median)}× slower` : m)]);
    rows.push(["vs eager", `${fmtRatio(A.ratio_vs_eager)}×`, `${fmtRatio(B.ratio_vs_eager)}×`, ""]);
    rows.push(["peak memory", fmtBytes(A.peak_bytes), fmtBytes(B.peak_bytes), pct(A.peak_bytes, B.peak_bytes)]);
    rows.push(["allocations", String(A.alloc_count ?? "–"), String(B.alloc_count ?? "–"), ""]);
  }
  for (const s of c.stages) rows.push([`ops · ${s.stage}`, String(s.a_ops ?? "–"), String(s.b_ops ?? "–"), s.a_ops !== s.b_ops ? pct(s.a_ops, s.b_ops) : ""]);
  return rows;
}
