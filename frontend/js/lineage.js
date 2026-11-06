// Pure helpers for the lineage view: group a stage's ops by origin line, test selections, build
// connector paths. Lineage is by source line (the finest identity the importer's locations carry).

const MAX_CHIPS = 14;

/**
 * Group displayable ops by the set of source lines they stand for.
 * Returns [{ key, lines, label, approximate, chips:[{name,count,approx,opIds}], hidden, total }]
 * ordered by first line; ops without a line form the trailing "unattributed" group (lines = []).
 */
export function groupOps(stageOps) {
  const groups = new Map();
  for (const o of stageOps?.ops ?? []) {
    if (!o.displayable) continue;
    const lines = [...(o.source_lines ?? [])].sort((a, b) => a - b);
    const key = lines.join("+") || "-";
    let g = groups.get(key);
    if (!g) groups.set(key, (g = { key, lines, label: lines.length ? lines.map((l) => `L${l}`).join("+") : "no line", chipMap: new Map(), total: 0, approximate: false }));
    let c = g.chipMap.get(o.name);
    if (!c) g.chipMap.set(o.name, (c = { name: o.name, count: 0, approx: false, opIds: [] }));
    c.count += 1; c.opIds.push(o.id); c.approx ||= o.approximate;
    g.total += 1; g.approximate ||= o.approximate;
  }
  return [...groups.values()]
    .map(({ chipMap, ...g }) => {
      const all = [...chipMap.values()];
      return { ...g, chips: all.slice(0, MAX_CHIPS), hidden: all.slice(MAX_CHIPS) };
    })
    .sort((a, b) => (a.lines.length ? a.lines[0] : Infinity) - (b.lines.length ? b.lines[0] : Infinity) || a.key.localeCompare(b.key));
}

/** Does a group stand for any selected source line? */
export const groupHit = (group, selLines) => selLines.length > 0 && group.lines.some((l) => selLines.includes(l));

/** Lines in the stage's stored snapshot text that print ops belonging to the selected source lines. */
export function highlightTextLines(stageOps, selLines) {
  if (!selLines.length) return [];
  const out = new Set();
  for (const o of stageOps?.ops ?? []) {
    if (o.text_line != null && (o.source_lines ?? []).some((l) => selLines.includes(l))) out.add(o.text_line);
  }
  return [...out].sort((a, b) => a - b);
}

/** Selection after clicking a group: its own lines (selecting again clears). */
export function toggleSelection(current, lines) {
  const same = current.length === lines.length && current.every((l, i) => l === lines[i]);
  return same || lines.length === 0 ? [] : [...lines];
}

/**
 * Orthogonal connector between the bottom of `from` and the top of `to` (rects with x,y,width,height
 * in the same coordinate space): down, across, down.
 */
export function connectorPath(from, to) {
  const x1 = Math.round(from.x + from.width / 2), y1 = Math.round(from.y + from.height);
  const x2 = Math.round(to.x + to.width / 2), y2 = Math.round(to.y);
  const mid = Math.round((y1 + y2) / 2);
  return x1 === x2 ? `M${x1} ${y1}V${y2}` : `M${x1} ${y1}V${mid}H${x2}V${y2}`;
}

/** Runtime share of a group (sum over its source lines) from metrics.per_line_share, or null. */
export function groupShare(group, perLine) {
  if (!perLine || !group.lines.length) return null;
  let s = 0, any = false;
  for (const l of group.lines) if (perLine[l] != null) { s += perLine[l]; any = true; }
  return any ? s : null;
}

/** Heat bucket 1..5 for a share (0 = none). */
export function heatBucket(share) {
  if (share == null || share <= 0) return 0;
  return share >= 0.4 ? 5 : share >= 0.2 ? 4 : share >= 0.1 ? 3 : share >= 0.03 ? 2 : 1;
}

/** Coverage text for a stage label. */
export function coverageText(pct) {
  return pct == null ? "" : `${Math.round(pct * 100)}% exact lineage`;
}
