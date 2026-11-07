// Single page: compile a model, read the result, trace any source line through every stage,
// walk the passes, inspect any snapshot.
import { api, watchRun } from "./api.js";
import { esc, fmtMs, pad3 } from "./format.js";
import { evidenceLine, evidenceRows, findingLines, impactText, indexLabel, stepIndex } from "./findings.js";
import { headlineModel } from "./headline.js";
import { compareRows, runLabel, treeOf } from "./history.js";
import { flavourForStage, renderCode } from "./ir.js";
import { connectorPath, coverageText, groupHit, groupOps, groupShare, heatBucket, highlightTextLines, toggleSelection } from "./lineage.js";
import { mountOrb, orbStateForPhase } from "./orb.js";
import { countUnit, describeDelta, opDelta, stageSnapshots } from "./passes.js";

const $ = (id) => document.getElementById(id);
const state = {
  templates: [], template: null, run: null, runId: null, progress: null, closeWatch: null,
  prov: null,               // { stage: StageOps } once the run is done
  selIdx: null, itab: "ir", iopen: false, busy: false, error: null,
  selFinding: null,         // id of the selected finding
  runs: [],                 // all runs (for the history chain)
  cmp: null,                // compare result when comparing two runs
  cmpDiff: null,            // stage whose IR diff is shown in the inspector while comparing
  selLines: [],             // source lines being traced
  openPasses: new Set(),    // stages whose pass list is expanded
};
const snapCache = new Map(); // `${runId}:${idx}` -> text

// thinking orbs: a 20px one by the status text and a 64px one while waiting for the first snapshots
let statusOrb = null, mapOrb = null;
function setStatusOrb(phase) {
  const el = $("orb");
  if (!phase) { el.hidden = true; statusOrb?.destroy(); statusOrb = null; return; }
  const next = orbStateForPhase(phase);
  el.hidden = false;
  if (statusOrb) statusOrb.set(next); else statusOrb = mountOrb(el, { state: next, size: 20 });
}
function setMapOrb(canvas, phase) {
  mapOrb?.destroy(); mapOrb = null;
  if (canvas) mapOrb = mountOrb(canvas, { state: orbStateForPhase(phase), size: 64 });
}

// ------------------------------------------------------------------ status + result bar

function renderStatus() {
  const el = $("status");
  el.className = "status";
  if (state.error) { setStatusOrb(null); el.classList.add("error"); el.textContent = state.error; return; }
  const run = state.run;
  if (!run) { setStatusOrb(state.busy ? "queued" : null); el.textContent = state.busy ? "starting…" : ""; return; }
  const m = headlineModel(run, state.progress);
  setStatusOrb(m.kind === "progress" ? m.phase : null);
  if (m.kind === "progress") el.textContent = [m.phase, m.step, m.name].filter(Boolean).join(" · ");
  else if (m.kind === "failed") { el.classList.add("error"); el.textContent = m.title; }
  else el.textContent = "";
}

function metric(label, value, unit, sub) {
  return `<div class="metric"><span class="kicker">${esc(label)}</span>
    <div class="value">${value}${unit ? `<small>${esc(unit)}</small>` : ""}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function renderMetrics() {
  const el = $("metrics");
  const run = state.run;
  if (!run) { el.hidden = true; return; }
  const m = headlineModel(run, state.progress);
  el.className = "metrics";
  if (m.kind === "progress") { el.hidden = true; return; }
  el.hidden = false;
  if (m.kind === "failed") {
    el.classList.add("failed");
    el.innerHTML = `<strong>${esc(m.title)}</strong>${m.lines.map((l) => `<span>${esc(l)}</span>`).join("")}`;
  } else if (m.kind === "mismatch") {
    el.classList.add("failed");
    el.innerHTML = `<strong>${esc(m.text)}</strong><span>speed numbers are hidden until the output matches PyTorch eager</span>`;
  } else {
    let html = metric("runtime", esc(m.runtime), "ms");
    if (m.vsEager) html += metric("vs eager", esc(m.vsEager.text.split(" ")[0]), "", `${esc(m.vsEager.text.split(" ").slice(1).join(" "))} (${esc(m.vsEager.threads)})`);
    else if (m.unverified) html += metric("vs eager", "–", "", "correctness not confirmed");
    if (m.hottest) {
      const share = m.hottest.share ? `${esc(m.hottest.share)} ` : "";
      html += metric("hottest", `L${m.hottest.line}`, "", `${share}${m.hottest.code ? `<code>${esc(m.hottest.code)}</code> ` : ""}· ${m.hottest.attribution}`);
    }
    el.innerHTML = html;
  }
}

// ------------------------------------------------------------------ findings

function renderFindings() {
  const run = state.run;
  const box = $("findings");
  box.hidden = !run || run.status !== "done";
  if (box.hidden) return;
  const fs = run.findings ?? [];
  $("findings-count").textContent = String(fs.length).padStart(2, "0");
  $("findings-list").innerHTML = fs.length ? fs.map((f, i) => {
    const sel = f.id === state.selFinding;
    const verified = f.impact_verified?.kind === "verified";
    const badge = sel && verified ? ` verified` : "";  // the accent marks exactly one thing: the selected finding's state
    const act = sel && f.fix_ids.length && !f.impact_verified
      ? `<span class="fx"><span class="fixbtn" role="button" tabindex="0" data-fix="${esc(f.id)}">fix <kbd>f</kbd></span></span>`
      : sel && f.child_run_id ? `<span class="fx"><span class="linkbtn" role="button" tabindex="0" data-open="${esc(f.child_run_id)}">open fixed run</span></span>` : "";
    return `<li><button class="finding${sel ? " sel" : ""}" data-fid="${esc(f.id)}" aria-pressed="${sel}"><span class="fi">${indexLabel(i)}</span>
      <span class="ft">${esc(f.title)}</span><span class="fv${badge}">${esc(impactText(f))}</span><span class="fe">${esc(evidenceLine(f))}${f.impact_verified?.note ? ` · ${esc(f.impact_verified.note)}` : ""}</span>${act}</button></li>`;
  }).join("")
    : `<li class="empty">no findings above threshold</li>`;
}

function selectFinding(id) {
  const f = state.run?.findings.find((x) => x.id === id);
  if (!f) return;
  state.selFinding = id;
  state.selLines = findingLines(f);
  state.itab = "evidence";
  state.iopen = true;
  renderFindings(); renderMap(); renderInspector();
}

function evidenceHtml(f) {
  const rows = evidenceRows(f).map(([k, kind, v]) => `<div class="ev"><span class="kicker">${esc(k.replaceAll("_", " "))}</span>${
    kind === "text" ? `<p>${esc(v)}</p>` : kind === "code" ? `<pre>${esc(v.join("\n"))}</pre>` : `<ul>${v.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`}</div>`).join("");
  return `<div class="evidence"><h3>${esc(f.title)}</h3><p class="sub">${esc(impactText(f))}${f.impact_est.note ? ` · ${esc(f.impact_est.note)}` : ""}${f.fix_ids.length ? ` · fix: ${esc(f.fix_ids.join(", "))}` : " · no automatic fix"}</p>${rows}</div>`;
}

// ------------------------------------------------------------------ run history + compare

function renderHistory() {
  const el = $("history");
  const chain = state.runId ? treeOf(state.runs, state.runId) : [];
  el.hidden = chain.length < 2;
  if (el.hidden) return;
  const parent = chain.find((r) => r.id === state.run?.parent_id);
  el.innerHTML = `<span class="kicker">runs</span><ol>${chain.map((r) => {
    const l = runLabel(r);
    return `<li><button class="run${r.id === state.runId ? " sel" : ""}${r.id === state.cmp?.b.id ? " cmp" : ""}" data-run="${r.id}" aria-pressed="${r.id === state.runId}"
      title="click to open · shift-click to compare with the open run"><span class="kicker">${esc(l.index)}</span> ${esc(l.name)} <b>${esc(l.time)}</b></button></li>`;
  }).join("")}</ol>${parent && state.run?.status === "done" ? `<button class="linkbtn" data-compare="${parent.id}">compare with ${esc(runLabel(parent).index)}</button>` : ""}`;
}

async function showCompare(otherId) {
  if (!state.runId || otherId === state.runId) return;
  try { state.cmp = await api.compare(state.runId, otherId); state.cmpDiff = null; } catch (e) { state.error = e.message; renderStatus(); return; }
  renderHistory(); renderMap(); renderInspector();
}

function closeCompare() { state.cmp = null; state.cmpDiff = null; renderHistory(); renderMap(); renderInspector(); }

function compareHtml(c) {
  const rows = compareRows(c).map(([k, a, b, d]) => `<tr><th scope="row">${esc(k)}</th><td>${esc(a)}</td><td>${esc(b)}</td><td>${esc(d)}</td></tr>`).join("");
  const stages = c.stages.filter((s) => s.a_snapshot_idx != null && s.b_snapshot_idx != null)
    .map((s) => `<button class="pass" data-cmpstage="${s.stage}">${esc(s.stage)}</button>`).join("");
  return `<div class="compare"><div class="cmp-head"><span class="kicker">compare · ${esc(runLabel(c.a).index)} ${esc(c.a.label || "run")} vs ${esc(runLabel(c.b).index)} ${esc(c.b.label || "run")}</span>
    <button class="linkbtn" data-closecmp>close</button></div>
    ${c.fix ? `<p class="sub">fix: ${esc(c.fix.label)} — ${esc(c.fix.description)}</p>` : ""}
    <table><thead><tr><th></th><th>${esc(runLabel(c.a).index)}</th><th>${esc(runLabel(c.b).index)}</th><th></th></tr></thead><tbody>${rows}</tbody></table>
    <p class="kicker">pipeline config diff</p>${diffHtml(c.config_diff)}
    <p class="kicker">ir diff per stage (final snapshot of each stage)</p><div class="passes">${stages}</div></div>`;
}

// ------------------------------------------------------------------ pipeline map

const linesAttr = (lines) => lines.join(",");
const hasSel = () => state.selLines.length > 0;

function sourceRows(run) {
  const perLine = run.metrics?.per_line_share;
  const groups = new Map();
  for (const n of run.fx_nodes) {
    if (n.op !== "call_function" || n.line == null) continue;
    const g = groups.get(n.line) ?? { code: n.code, nodes: [] };
    g.nodes.push(n.name);
    groups.set(n.line, g);
  }
  if (!groups.size) return `<p class="readout">no source lines recorded</p>`;
  return `<div class="src-lines${hasSel() ? " sel" : ""}">${[...groups.entries()].sort((a, b) => a[0] - b[0]).map(([line, g]) =>
    `<button class="src-line${state.selLines.includes(line) ? " hit" : ""}" data-lines="${line}" data-stage="source" title="trace L${line} through every stage">
      <span class="ln">L${line}</span><span class="code">${esc(g.code ?? "")}</span><span class="nodes">${perLine?.[line] != null ? `${isEst() ? "~" : ""}${pct(perLine[line])} · ` : ""}${esc(g.nodes.join(" · "))}</span></button>`).join("")}</div>`;
}

const pct = (s) => (s < 0.005 ? "<1%" : `${Math.round(s * 100)}%`);
const isEst = () => state.run?.metrics?.attribution === "est";

function groupsHtml(stageId, sops) {
  const perLine = state.run?.metrics?.per_line_share;
  return `<div class="groups">${groupOps(sops).map((g) => {
    const hit = groupHit(g, state.selLines);
    const cls = `grp${hit ? " hit" : ""}${hasSel() && !hit ? " dimmed" : ""}`;
    const chips = g.chips.map((c) => `<button class="chip${c.approx ? " approx" : ""}" data-lines="${linesAttr(g.lines)}" data-stage="${stageId}"
        title="${esc(`${c.name} ×${c.count} · ${g.lines.length ? `from ${g.label}` : "no source line"}${c.approx ? " · approximate (inferred from dataflow)" : ""}`)}">${esc(c.name)}${c.count > 1 ? `<b>×${c.count}</b>` : ""}</button>`).join("");
    const more = g.hidden.length ? `<span class="more" title="${esc(g.hidden.map((c) => `${c.name} ×${c.count}`).join(", "))}">+${g.hidden.length} more</span>` : "";
    const share = groupShare(g, perLine);
    const heat = heatBucket(share);
    const shareTxt = share == null ? "" : ` ${isEst() ? "~" : ""}${pct(share)}`;
    return `<div class="${cls}" data-lines="${linesAttr(g.lines)}" data-approx="${g.approximate ? 1 : 0}">
      <button class="gl${heat ? ` heat${isEst() ? " est" : ""}` : ""}" ${heat ? `style="--h:var(--heat-${heat})" ` : ""}data-lines="${linesAttr(g.lines)}" data-stage="${stageId}"${g.lines.length ? ` title="${esc(`trace ${g.label}${share == null ? "" : ` · ${pct(share)} of runtime (${isEst() ? "estimate" : "measured"})`}`)}"` : " disabled"}>${esc(g.label)}${shareTxt}</button>
      <span class="chips">${chips}${more}</span></div>`;
  }).join("")}</div>`;
}

function passBox(stage, items) {
  const changed = items.filter((i) => i.snap.changed).length;
  const open = state.openPasses.has(stage.id) || !state.prov;
  return `<details class="passbox" data-stage="${stage.id}"${open ? " open" : ""}>
    <summary>passes · ${items.length} · ${changed} changed</summary>
    <ul class="passes" aria-label="${esc(stage.name)} passes">${items.map(({ snap }) =>
      `<li><button class="pass${snap.changed ? " changed" : ""}${snap.composite ? " composite" : ""}${snap.idx === state.selIdx ? " sel" : ""}"
        data-idx="${snap.idx}" aria-pressed="${snap.idx === state.selIdx}"${snap.composite ? ' title="part of a composite pipeline"' : ""}>
        <i aria-hidden="true"></i><span class="sr-only">${snap.changed ? "changed: " : "unchanged: "}</span>${esc(snap.pass)}</button></li>`).join("")}</ul>
    <p class="readout" data-readout="${stage.id}" aria-live="polite"></p></details>`;
}

function renderMap() {
  const el = $("map");
  const run = state.run;
  if (state.cmp) { setMapOrb(null, null); el.innerHTML = compareHtml(state.cmp); return; }
  if (!run) { setMapOrb(null, null); el.innerHTML = `<p class="empty">Compile a model to see every pass it goes through.</p>`; $("map-count").textContent = ""; return; }
  const scroll = el.scrollTop;
  const rows = run.stages.filter((s) => s.id === "source" || s.snapshot_idxs.length);
  $("map-count").innerHTML = run.snapshots.length ? `${run.snapshots.length} snapshots${state.prov ? ` · <span class="legend">dashed = approximate</span>` : ""}` : "";
  if (!rows.length) {
    if (run.status === "error") { setMapOrb(null); el.innerHTML = `<p class="empty">No stages were produced.</p>`; return; }
    const phase = state.progress?.phase ?? run.phase;
    el.innerHTML = `<div class="waiting"><canvas></canvas><p>${esc(phase)}…</p></div>`;
    setMapOrb(el.querySelector("canvas"), phase);
    return;
  }
  setMapOrb(null, null);
  el.innerHTML = rows.map((stage) => {
    const items = stageSnapshots(run.snapshots, stage.snapshot_idxs);
    const last = items.at(-1)?.snap;
    const sops = state.prov?.[stage.id];
    let meta, body;
    if (stage.id === "source") {
      meta = `${run.fx_nodes.filter((n) => n.op === "call_function").length} fx nodes`;
      body = sourceRows(run);
    } else {
      meta = `${items.length} ${items.length === 1 ? "snapshot" : "snapshots"}${last ? ` · ${last.total_ops} ${countUnit(stage.id)}` : ""}`;
      body = (sops ? groupsHtml(stage.id, sops) : "") + passBox(stage, items);
    }
    const cov = stage.id === "source" ? "" : coverageText(stage.exact_lineage_pct ?? sops?.exact_lineage_pct);
    return `<div class="stage" data-stage="${stage.id}"><div class="stage-label"><span class="kicker">${esc(stage.name)}</span>
      <p class="meta">${esc(meta)}</p>${cov ? `<p class="meta" title="share of ops whose source line comes from a recorded location, not a heuristic">${esc(cov)}</p>` : ""}</div>
      <div class="stage-body">${body}</div></div>`;
  }).join("");
  el.scrollTop = scroll;
  drawConnectors();
}

/** Solid line = exact, dashed = approximate. Drawn only for the traced lines, between adjacent rows. */
function drawConnectors() {
  const map = $("map");
  map.querySelector("svg.links")?.remove();
  if (!hasSel()) return;
  const mr = map.getBoundingClientRect();
  const rect = (el) => { const r = el.getBoundingClientRect(); return { x: r.left - mr.left + map.scrollLeft, y: r.top - mr.top + map.scrollTop, width: r.width, height: r.height }; };
  const rows = [...map.querySelectorAll(".stage")];
  const paths = [];
  for (let i = 0; i + 1 < rows.length; i++) {
    const upper = [...rows[i].querySelectorAll(".grp.hit, .src-line.hit")];
    const lower = [...rows[i + 1].querySelectorAll(".grp.hit")];
    for (const b of lower) {
      const bl = b.dataset.lines.split(",").map(Number);
      for (const a of upper) {
        const al = (a.dataset.lines ?? "").split(",").map(Number);
        if (!al.some((l) => bl.includes(l))) continue;
        const approx = a.dataset.approx === "1" || b.dataset.approx === "1";
        const A = rect(a), B = rect(b);
        paths.push({ d: connectorPath(A, B), approx, mx: (A.x + A.width / 2 + B.x + B.width / 2) / 2, my: (A.y + A.height + B.y) / 2 });
      }
    }
  }
  if (!paths.length) return;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "links");
  svg.setAttribute("width", map.scrollWidth);
  svg.setAttribute("height", map.scrollHeight);
  svg.setAttribute("aria-hidden", "true");
  for (const p of paths) {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", p.d);
    if (p.approx) path.setAttribute("class", "approx");
    svg.appendChild(path);
    if (p.approx) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", p.mx + 4); t.setAttribute("y", p.my); t.textContent = "approx";
      svg.appendChild(t);
    }
  }
  map.appendChild(svg);
}

function readoutFor(idx) {
  const run = state.run;
  const snap = run?.snapshots.find((s) => s.idx === idx);
  if (!snap) return "";
  const prev = run.snapshots.find((s) => s.idx === idx - 1);
  return `${snap.pass} · ${fmtMs(snap.time_ms)} ms · ${snap.changed ? "changed" : "unchanged"} · ${describeDelta(opDelta(prev, snap), snap.stage)}`;
}

// ------------------------------------------------------------------ inspector

async function snapshotText(runId, idx) {
  const key = `${runId}:${idx}`;
  if (!snapCache.has(key)) snapCache.set(key, await api.snapshot(runId, idx));
  return snapCache.get(key);
}

function diffHtml(unified) {
  if (!unified) return `<p class="empty">The two snapshots are identical.</p>`;
  const lines = unified.split("\n");
  if (lines.at(-1) === "") lines.pop();
  return `<pre class="diff">${lines.map((l) => {
    const cls = l.startsWith("@@") ? "hunk" : l.startsWith("+++") || l.startsWith("---") ? "meta" : l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : "";
    return `<span${cls ? ` class="${cls}"` : ""}>${esc(l)}</span>`;
  }).join("")}</pre>`;
}

let inspectorToken = 0;
async function renderInspector() {
  const body = $("inspector-body");
  const run = state.run;
  $("inspector-toggle").textContent = state.iopen ? "collapse" : "open";
  $("inspector-toggle").setAttribute("aria-expanded", String(state.iopen));
  document.querySelectorAll("[data-itab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.itab === state.itab)));
  const snap = run?.snapshots.find((s) => s.idx === state.selIdx);
  const sops = snap && state.prov?.[snap.stage];
  const traced = sops && sops.snapshot_idx === snap.idx ? highlightTextLines(sops, state.selLines) : [];
  $("inspector-title").textContent = snap
    ? `${snap.stage} · ${pad3(snap.idx)} ${snap.pass}${snap.composite ? " · composite" : ""}${traced.length ? ` · ${traced.length} lines traced` : ""}` : "";
  body.hidden = !state.iopen;
  if (!state.iopen) return;
  if (state.cmp && state.cmpDiff) {
    const s = state.cmp.stages.find((x) => x.stage === state.cmpDiff);
    $("inspector-title").textContent = `compare · ${s.stage} · ${state.cmp.a.id} → ${state.cmp.b.id}`;
    body.hidden = false;
    body.innerHTML = `<p class="empty">loading…</p>`;
    try {
      const d = await api.diff(state.cmp.a.id, s.a_snapshot_idx, s.b_snapshot_idx, state.cmp.b.id);
      body.innerHTML = diffHtml(d.unified);
    } catch (e) { body.innerHTML = `<p class="empty" style="color:var(--error)">${esc(e.message)}</p>`; }
    return;
  }
  if (state.itab === "evidence") {
    const f = run?.findings.find((x) => x.id === state.selFinding);
    body.innerHTML = f ? evidenceHtml(f) : `<p class="empty">Select a finding to see its evidence.</p>`;
    return;
  }
  if (!snap) { body.innerHTML = `<p class="empty">Select a pass or an op in the pipeline map to see its IR.</p>`; return; }

  const token = ++inspectorToken;
  body.innerHTML = `<p class="empty">loading…</p>`;
  try {
    if (state.itab === "ir") {
      const text = await snapshotText(run.id, snap.idx);
      if (token !== inspectorToken) return;
      const { gutter, html } = renderCode(flavourForStage(snap.stage), text, traced);
      body.innerHTML = `<div class="ir"><div class="gutter" aria-hidden="true">${gutter}</div><pre class="text" tabindex="0" aria-label="${esc(snap.stage)} IR after ${esc(snap.pass)}">${html}</pre></div>`;
      const first = body.querySelector(".l.hl");
      if (first) body.scrollTop = Math.max(0, first.offsetTop - body.clientHeight / 2);
    } else if (snap.idx === 0) {
      body.innerHTML = `<p class="empty">This is the first snapshot: there is nothing before it to compare with.</p>`;
    } else {
      const d = await api.diff(run.id, snap.idx - 1, snap.idx);
      if (token !== inspectorToken) return;
      body.innerHTML = diffHtml(d.unified);
    }
  } catch (e) {
    if (token === inspectorToken) body.innerHTML = `<p class="empty" style="color:var(--error)">${esc(e.message)}</p>`;
  }
}

function selectSnapshot(idx) {
  state.selIdx = idx;
  state.iopen = true;
  const stage = state.run?.snapshots.find((s) => s.idx === idx)?.stage;
  if (stage) state.openPasses.add(stage);
  renderMap();
  renderInspector();
}

/** Trace source lines through every stage; show the clicked stage's final IR with those lines marked. */
function traceLines(lines, stageId) {
  state.selLines = toggleSelection(state.selLines, lines);
  const so = state.prov?.[stageId];
  if (hasSel() && so?.snapshot_idx != null) {
    state.selIdx = so.snapshot_idx;
    state.itab = "ir";
    state.iopen = true;
  }
  renderMap();
  renderInspector();
}

// ------------------------------------------------------------------ run lifecycle

function renderAll() { renderStatus(); renderMetrics(); renderHistory(); renderFindings(); renderMap(); renderInspector(); }

async function loadProvenance() {
  const run = state.run;
  if (!run || run.status !== "done" || state.prov) return;
  const ids = run.stages.filter((s) => s.id === "source" || s.snapshot_idxs.length).map((s) => s.id);
  try {
    const all = await Promise.all(ids.map((id) => api.stage(run.id, id)));
    if (state.runId === run.id) state.prov = Object.fromEntries(all.map((s) => [s.stage, s]));
  } catch { /* the map still works without lineage */ }
}

let fetchTimer = null;
function scheduleFetch() {
  if (fetchTimer) return;
  fetchTimer = setTimeout(async () => { fetchTimer = null; await refreshRun(); }, 400);
}

async function refreshRun() {
  if (!state.runId) return;
  try {
    state.run = await api.run(state.runId);
    await loadProvenance();
    renderAll();
  } catch (e) { state.error = e.message; renderStatus(); }
}

/** Show a run (new, child or from the history); follow its progress if it is still compiling. */
async function openRun(run_id) {
  state.closeWatch?.();
  Object.assign(state, { runId: run_id, run: null, progress: null, prov: null, selIdx: null, iopen: false, selLines: [], selFinding: null, cmp: null, cmpDiff: null });
  state.openPasses.clear();
  await refreshRuns();
  await refreshRun();
  if (state.run && (state.run.status === "queued" || state.run.status === "running")) {
    state.busy = true; $("compile").disabled = true;
    state.closeWatch = watchRun(run_id, (ev) => { state.progress = ev; renderStatus(); scheduleFetch(); }, async () => {
      await refreshRun(); await refreshRuns(); renderAll();
      state.busy = false; $("compile").disabled = false;
    });
  }
}

async function refreshRuns() { try { state.runs = await api.runs(); } catch { /* history is optional */ } }

async function compile() {
  if (state.busy) return;
  const source = $("source").value;
  state.error = null;
  try {
    const sameAsTemplate = state.template && source === state.template.source;
    const { run_id } = await api.createRun({ source, template: sameAsTemplate ? state.template.name : null });
    snapCache.clear();
    await openRun(run_id);
  } catch (e) { state.error = e.message; state.busy = false; $("compile").disabled = false; renderStatus(); }
}

async function applyFix(findingId) {
  if (state.busy || !state.runId) return;
  try {
    const { run_id } = await api.applyFix(state.runId, findingId);
    snapCache.clear();
    await openRun(run_id);
  } catch (e) { state.error = e.message; renderStatus(); }
}

// ------------------------------------------------------------------ setup

async function loadTemplate(name) {
  state.template = await api.template(name);
  $("source").value = state.template.source;
}

function moveStage(dir) {
  const run = state.run;
  if (!run) return;
  const stages = run.stages.filter((s) => s.id !== "source" && s.snapshot_idxs.length);
  if (!stages.length) return;
  const curStage = run.snapshots.find((s) => s.idx === state.selIdx)?.stage;
  let i = stages.findIndex((s) => s.id === curStage);
  i = dir > 0 ? Math.min(stages.length - 1, i + 1) : Math.max(0, i === -1 ? 0 : i - 1);
  selectSnapshot(stages[i].snapshot_idxs.at(-1));
}

async function boot() {
  try {
    state.templates = await api.templates();
    $("template").innerHTML = state.templates.map((t) => `<option value="${esc(t.name)}" title="${esc(t.description)}">${esc(t.name)}</option>`).join("");
    const first = state.templates.find((t) => t.name === "attention_block") ?? state.templates[0];
    if (first) { $("template").value = first.name; await loadTemplate(first.name); }
  } catch (e) { state.error = `could not reach the API: ${e.message}`; renderStatus(); }
  api.toolchain().then((t) => {
    $("toolchain").textContent = `llvm ${t.system_llvm ?? "missing"} · mlir ${t.bundled_mlir_llvm ?? "?"}`;
    $("toolchain").title = (t.warnings ?? []).join("\n");
    if (!t.system_llvm) { state.error = (t.warnings ?? ["LLVM tools not found"])[0]; renderStatus(); }
  }).catch(() => {});

  $("template").addEventListener("change", (e) => loadTemplate(e.target.value).catch((err) => { state.error = err.message; renderStatus(); }));
  $("compile").addEventListener("click", compile);
  $("inspector-toggle").addEventListener("click", () => { state.iopen = !state.iopen; renderInspector(); });
  document.querySelectorAll("[data-itab]").forEach((b) => b.addEventListener("click", () => { state.itab = b.dataset.itab; state.iopen = true; renderInspector(); }));

  $("findings-list").addEventListener("click", (e) => {
    const fix = e.target.closest("[data-fix]"), open = e.target.closest("[data-open]");
    if (fix) { e.stopPropagation(); applyFix(fix.dataset.fix); return; }
    if (open) { e.stopPropagation(); openRun(open.dataset.open); return; }
    const b = e.target.closest(".finding"); if (b) selectFinding(b.dataset.fid);
  });
  $("history").addEventListener("click", (e) => {
    const cmpBtn = e.target.closest("[data-compare]"), run = e.target.closest("[data-run]");
    if (cmpBtn) showCompare(cmpBtn.dataset.compare);
    else if (run) { if (e.shiftKey) showCompare(run.dataset.run); else openRun(run.dataset.run); }
  });
  $("map").addEventListener("click", (e) => {
    if (e.target.closest("[data-closecmp]")) { closeCompare(); return; }
    const st = e.target.closest("[data-cmpstage]");
    if (st) { state.cmpDiff = st.dataset.cmpstage; state.iopen = true; renderInspector(); }
  });
  $("map").addEventListener("click", (e) => {
    const pass = e.target.closest(".pass");
    if (pass) { selectSnapshot(Number(pass.dataset.idx)); return; }
    const t = e.target.closest(".chip, .gl, .src-line");
    if (t && t.dataset.lines) traceLines(t.dataset.lines.split(",").map(Number), t.dataset.stage);
  });
  $("map").addEventListener("toggle", (e) => {
    const d = e.target;
    if (d.classList?.contains("passbox")) { d.open ? state.openPasses.add(d.dataset.stage) : state.openPasses.delete(d.dataset.stage); drawConnectors(); }
  }, true);
  window.addEventListener("resize", drawConnectors);

  const readout = (e, on) => {
    const b = e.target.closest?.(".pass");
    if (!b) return;
    const el = b.closest(".passbox")?.querySelector("[data-readout]");
    if (el) el.textContent = on ? readoutFor(Number(b.dataset.idx)) : "";
  };
  $("map").addEventListener("mouseover", (e) => readout(e, true));
  $("map").addEventListener("mouseout", (e) => readout(e, false));
  $("map").addEventListener("focusin", (e) => readout(e, true));
  $("map").addEventListener("focusout", (e) => readout(e, false));

  $("source").addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      const t = e.target, s = t.selectionStart;
      t.value = t.value.slice(0, s) + "    " + t.value.slice(t.selectionEnd);
      t.selectionStart = t.selectionEnd = s + 4;
    }
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); compile(); return; }
    const tag = e.target.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "]") { e.preventDefault(); moveStage(1); }
    if (e.key === "[") { e.preventDefault(); moveStage(-1); }
    if (e.key === "f" && state.selFinding) { e.preventDefault(); applyFix(state.selFinding); }
    if (e.key === "Escape" && state.cmp) { closeCompare(); return; }
    if (e.key === "Escape" && hasSel()) { state.selLines = []; state.selFinding = null; renderFindings(); renderMap(); renderInspector(); }
    if ((e.key === "j" || e.key === "k") && state.run?.findings?.length) {
      const fs = state.run.findings;
      selectFinding(fs[stepIndex(fs.findIndex((f) => f.id === state.selFinding), e.key === "j" ? 1 : -1, fs.length)].id);
    }
  });
  renderAll();
}

boot();
