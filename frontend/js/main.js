// Single page: compile a model, read the result, walk the passes, inspect any snapshot.
import { api, watchRun } from "./api.js";
import { esc, fmtMs, pad3 } from "./format.js";
import { headlineModel } from "./headline.js";
import { flavourForStage, renderCode } from "./ir.js";
import { countUnit, describeDelta, opDelta, stageSnapshots } from "./passes.js";

const $ = (id) => document.getElementById(id);
const state = {
  templates: [], template: null, run: null, runId: null, progress: null, closeWatch: null,
  selIdx: null, itab: "ir", iopen: false, busy: false, error: null,
};
const snapCache = new Map(); // `${runId}:${idx}` -> text

// ------------------------------------------------------------------ status + result bar

function renderStatus() {
  const el = $("status");
  const run = state.run;
  el.className = "status";
  if (state.error) { el.classList.add("error"); el.textContent = state.error; return; }
  if (!run) { el.textContent = ""; return; }
  const m = headlineModel(run, state.progress);
  if (m.kind === "progress") {
    el.textContent = [m.phase, m.step, m.name].filter(Boolean).join(" · ");
  } else if (m.kind === "failed") {
    el.classList.add("error");
    el.textContent = m.title;
  } else {
    el.textContent = "";
  }
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

// ------------------------------------------------------------------ pipeline map

function sourceRows(run) {
  const groups = new Map();
  for (const n of run.fx_nodes) {
    if (n.op !== "call_function" || n.line == null) continue;
    const g = groups.get(n.line) ?? { code: n.code, nodes: [] };
    g.nodes.push(n.name);
    groups.set(n.line, g);
  }
  if (!groups.size) return `<p class="readout">no source lines recorded</p>`;
  return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([line, g]) =>
    `<div class="src-line"><span class="ln">L${line}</span><span class="code">${esc(g.code ?? "")}</span><span class="nodes">${esc(g.nodes.join(" · "))}</span></div>`).join("");
}

function renderMap() {
  const el = $("map");
  const run = state.run;
  if (!run) { el.innerHTML = `<p class="empty">Compile a model to see every pass it goes through.</p>`; $("map-count").textContent = ""; return; }
  const scroll = el.scrollTop;
  const rows = run.stages.filter((s) => s.id === "source" || s.snapshot_idxs.length);
  $("map-count").textContent = run.snapshots.length ? `${run.snapshots.length} snapshots` : "";
  if (!rows.length) {
    el.innerHTML = `<p class="empty">${run.status === "error" ? "No stages were produced." : "Waiting for the first snapshots…"}</p>`;
    return;
  }
  el.innerHTML = rows.map((stage) => {
    const items = stageSnapshots(run.snapshots, stage.snapshot_idxs);
    const last = items.at(-1)?.snap;
    let meta, body;
    if (stage.id === "source") {
      meta = `${run.fx_nodes.filter((n) => n.op === "call_function").length} fx nodes`;
      body = sourceRows(run);
    } else {
      meta = `${items.length} ${items.length === 1 ? "snapshot" : "snapshots"}${last ? ` · ${last.total_ops} ${countUnit(stage.id)}` : ""}`;
      body = `<ul class="passes" aria-label="${esc(stage.name)} passes">${items.map(({ snap }) =>
        `<li><button class="pass${snap.changed ? " changed" : ""}${snap.composite ? " composite" : ""}${snap.idx === state.selIdx ? " sel" : ""}"
          data-idx="${snap.idx}" aria-pressed="${snap.idx === state.selIdx}"${snap.composite ? ' title="part of a composite pipeline"' : ""}>
          <i aria-hidden="true"></i><span class="sr-only">${snap.changed ? "changed: " : "unchanged: "}</span>${esc(snap.pass)}</button></li>`).join("")}</ul>
        <p class="readout" data-readout="${stage.id}" aria-live="polite"></p>`;
    }
    return `<div class="stage" data-stage="${stage.id}"><div class="stage-label"><span class="kicker">${esc(stage.name)}</span><p class="meta">${esc(meta)}</p></div><div class="stage-body">${body}</div></div>`;
  }).join("");
  el.scrollTop = scroll;
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
  $("inspector-title").textContent = snap ? `${snap.stage} · ${pad3(snap.idx)} ${snap.pass}${snap.composite ? " · composite" : ""}` : "";
  body.hidden = !state.iopen;
  if (!state.iopen) return;
  if (!snap) { body.innerHTML = `<p class="empty">Select a pass in the pipeline map to see its IR.</p>`; return; }
  if (state.itab === "evidence") { body.innerHTML = `<p class="empty">Evidence appears here when a finding is selected.</p>`; return; }

  const token = ++inspectorToken;
  body.innerHTML = `<p class="empty">loading…</p>`;
  try {
    if (state.itab === "ir") {
      const text = await snapshotText(run.id, snap.idx);
      if (token !== inspectorToken) return;
      const { gutter, html } = renderCode(flavourForStage(snap.stage), text);
      body.innerHTML = `<div class="ir"><pre class="gutter" aria-hidden="true">${gutter}</pre><pre class="text" tabindex="0" aria-label="${esc(snap.stage)} IR after ${esc(snap.pass)}">${html}</pre></div>`;
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
  renderMap();
  renderInspector();
}

// ------------------------------------------------------------------ run lifecycle

function renderAll() { renderStatus(); renderMetrics(); renderMap(); renderInspector(); }

let fetchTimer = null;
function scheduleFetch() {
  if (fetchTimer) return;
  fetchTimer = setTimeout(async () => { fetchTimer = null; await refreshRun(); }, 400);
}

async function refreshRun() {
  if (!state.runId) return;
  try { state.run = await api.run(state.runId); renderAll(); } catch (e) { state.error = e.message; renderStatus(); }
}

async function compile() {
  if (state.busy) return;
  const source = $("source").value;
  state.busy = true; state.error = null; $("compile").disabled = true;
  try {
    const sameAsTemplate = state.template && source === state.template.source;
    const { run_id } = await api.createRun({ source, template: sameAsTemplate ? state.template.name : null });
    state.closeWatch?.();
    Object.assign(state, { runId: run_id, run: null, progress: null, selIdx: null, iopen: false });
    snapCache.clear();
    await refreshRun();
    state.closeWatch = watchRun(run_id, (ev) => { state.progress = ev; renderStatus(); scheduleFetch(); }, async () => {
      await refreshRun();
      state.busy = false; $("compile").disabled = false;
    });
  } catch (e) {
    state.error = e.message; state.busy = false; $("compile").disabled = false; renderStatus();
  }
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
  $("map").addEventListener("click", (e) => { const b = e.target.closest(".pass"); if (b) selectSnapshot(Number(b.dataset.idx)); });
  const readout = (e, on) => {
    const b = e.target.closest?.(".pass");
    if (!b) return;
    const el = b.closest(".stage")?.querySelector("[data-readout]");
    if (el) el.textContent = on ? readoutFor(Number(b.dataset.idx)) : state.selIdx != null && b.closest(".stage").contains(document.querySelector(".pass.sel")) ? readoutFor(state.selIdx) : "";
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
  });
  renderAll();
}

boot();
