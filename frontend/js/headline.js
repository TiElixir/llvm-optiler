// State logic of the result bar, kept pure so it can be tested without a DOM.
import { fmtMs, fmtPct, fmtRatio } from "./format.js";

export const pluralThreads = (n) => `${n} thread${n === 1 ? "" : "s"}`;

function fmtErr(x) {
  if (x == null || !Number.isFinite(x)) return "n/a";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a < 1e-3 || a >= 1e4 ? x.toExponential(2) : x.toPrecision(3);
}

function vsEager(ratio, threads) {
  if (ratio == null || !Number.isFinite(ratio) || ratio <= 0) return null;
  const t = pluralThreads(threads);
  if (fmtRatio(ratio) === "1.00") return { ratio: "1.00", direction: "same", text: "on par with eager", threads: t };
  const direction = ratio > 1 ? "slower" : "faster"; // ratio = compiled / eager
  const shown = fmtRatio(ratio > 1 ? ratio : 1 / ratio);
  return { ratio: shown, direction, text: `${shown}× ${direction} than eager`, threads: t };
}

/**
 * @returns one of
 *  {kind:"progress", phase, step, name} | {kind:"failed", title, lines} |
 *  {kind:"mismatch", text} | {kind:"ok", runtime, vsEager, unverified, hottest}
 * A speed claim needs `correct === true`; a mismatch hides every number.
 */
export function headlineModel(run, progress) {
  if (run.status === "queued" || run.status === "running") {
    const phase = progress?.phase ?? run.phase;
    const n = progress?.n ?? null, total = progress?.total ?? null;
    return { kind: "progress", phase, step: n != null && total ? `${n}/${total}` : null, name: progress?.name ?? null };
  }
  if (run.status === "error") {
    const lines = (run.error ?? "").split("\n").map((l) => l.trimEnd()).filter((l) => l.trim()).slice(0, 3);
    return { kind: "failed", title: run.error_stage ? `compile failed at ${run.error_stage}` : "compile failed", lines };
  }
  const m = run.metrics;
  if (m.correct === false) {
    const err = m.max_abs_err != null ? ` (max abs err ${fmtErr(m.max_abs_err)})` : "";
    return { kind: "mismatch", text: `output mismatch at native build${err}` };
  }
  const verified = m.correct === true;
  const hottest = m.hottest_line == null ? null : {
    share: m.hottest_line_share == null ? null : fmtPct(m.hottest_line_share),
    line: m.hottest_line, code: m.hottest_line_code,
    attribution: m.attribution === "est" ? "est" : "measured",
  };
  return {
    kind: "ok", runtime: fmtMs(m.runtime_ms_median),
    vsEager: verified ? vsEager(m.ratio_vs_eager, m.threads) : null,
    unverified: !verified, hottest,
  };
}
