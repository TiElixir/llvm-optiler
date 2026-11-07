// Advanced drawer: pure helpers (tested).

export const TRANSFORM_PATH = "${RUN_DIR}/transforms/user.mlir";
const PRELOAD = `transform-preload-library{transform-library-paths=${TRANSFORM_PATH}}`;
const INTERP = "transform-interpreter";

/**
 * Append the two passes that run a user transform script to the end of the linalg_tensor stage of a pipeline
 * YAML (`- stage: x` / `passes:` / `- pass`). Returns the text unchanged if they are already there or the
 * stage is missing.
 */
export function attachTransform(yaml) {
  if (yaml.includes(PRELOAD)) return yaml;
  const lines = yaml.split("\n");
  const start = lines.findIndex((l) => /^-\s*stage:\s*linalg_tensor\s*$/.test(l));
  if (start < 0) return yaml;
  let end = lines.findIndex((l, i) => i > start && /^-\s*stage:/.test(l));
  if (end < 0) end = lines.length;
  while (end > start + 1 && lines[end - 1].trim() === "") end--;
  const last = lines.slice(start, end).reverse().find((l) => /^\s*-\s/.test(l));
  const indent = last ? last.match(/^(\s*)-/)[1] : "    ";
  lines.splice(end, 0, `${indent}- "${PRELOAD}"`, `${indent}- ${INTERP}`);
  return lines.join("\n");
}

/** Validate settings inputs; returns { ok, values, error }. */
export function readSettings({ iters, warmup, threads }) {
  const v = { iters: Number(iters), warmup: Number(warmup), threads: Number(threads) };
  if (!Number.isInteger(v.iters) || v.iters < 1 || v.iters > 10000) return { ok: false, error: "iterations must be a whole number from 1 to 10000" };
  if (!Number.isInteger(v.warmup) || v.warmup < 0 || v.warmup > 1000) return { ok: false, error: "warmup must be a whole number from 0 to 1000" };
  if (!Number.isInteger(v.threads) || v.threads < 1 || v.threads > 256) return { ok: false, error: "threads must be a whole number from 1 to 256" };
  return { ok: true, values: v };
}

/** Rows for the per-pass table, slowest flagged. */
export function passRows(snapshots) {
  const times = snapshots.map((s) => s.time_ms).sort((a, b) => b - a);
  const cut = times[Math.min(4, times.length - 1)] ?? 0;
  return snapshots.map((s) => ({ idx: s.idx, stage: s.stage, pass: s.pass, changed: s.changed, ms: s.time_ms, slow: s.time_ms > 0 && s.time_ms >= cut }));
}
