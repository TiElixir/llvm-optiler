// Per-pass helpers for the pipeline map.

/** Op-count change between two snapshots (names whose count differs, largest change first). */
export function opDelta(prev, cur) {
  const before = prev?.op_counts ?? {};
  const after = cur.op_counts ?? {};
  const names = new Set([...Object.keys(before), ...Object.keys(after)]);
  const changed = [];
  for (const n of names) {
    const a = before[n] ?? 0, b = after[n] ?? 0;
    if (a !== b) changed.push([n, a, b]);
  }
  changed.sort((x, y) => Math.abs(y[2] - y[1]) - Math.abs(x[2] - x[1]) || x[0].localeCompare(y[0]));
  return { total: [prev?.total_ops ?? 0, cur.total_ops], changed };
}

/** `total_ops` counts ops for MLIR snapshots and lines for LLVM IR / asm. */
export const countUnit = (stage) => (stage === "llvm_ir" || stage === "asm" ? "lines" : "ops");

export function describeDelta(d, stage, max = 4) {
  if (countUnit(stage) === "lines") return `${d.total[0]} → ${d.total[1]} lines`;
  const head = `${d.total[0]} → ${d.total[1]} ops`;
  if (!d.changed.length) return `${head}, no op-count change`;
  const rest = d.changed.slice(0, max).map(([n, a, b]) => `${n} ${a} → ${b}`).join(", ");
  const more = d.changed.length > max ? `, +${d.changed.length - max} more` : "";
  return `${head}: ${rest}${more}`;
}

/** A stage's snapshots in order, each with the snapshot that precedes it in the run. */
export function stageSnapshots(snapshots, idxs) {
  const byIdx = new Map(snapshots.map((s) => [s.idx, s]));
  return idxs.map((i) => byIdx.get(i)).filter(Boolean).map((snap) => ({ snap, prev: byIdx.get(snap.idx - 1) }));
}
