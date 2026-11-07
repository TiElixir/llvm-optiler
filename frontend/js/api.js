// Thin wrappers over the HTTP API (same origin).

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { const b = await res.json(); detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  templates: () => fetch("/api/templates").then(json),
  template: (name) => fetch(`/api/templates/${encodeURIComponent(name)}`).then(json),
  toolchain: () => fetch("/api/toolchain").then(json),
  run: (id) => fetch(`/api/runs/${id}`).then(json),
  createRun: (req) => fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) }).then(json),
  snapshot: (id, idx) => fetch(`/api/runs/${id}/snapshots/${idx}`).then(async (r) => { if (!r.ok) throw new Error(r.statusText); return r.text(); }),
  stage: (id, stage) => fetch(`/api/runs/${id}/stages/${stage}`).then(json),
  runs: () => fetch("/api/runs").then(json),
  applyFix: (id, fid) => fetch(`/api/runs/${id}/findings/${encodeURIComponent(fid)}/apply`, { method: "POST" }).then(json),
  compare: (a, b) => fetch(`/api/runs/${a}/compare?other=${b}`).then(json),
  diff: (id, a, b, bRun) => fetch(`/api/runs/${id}/diff?a=${a}&b=${b}${bRun ? `&b_run=${bRun}` : ""}`).then(json),
};

/** Follow a run's progress stream. Returns a function that closes it. */
export function watchRun(runId, onEvent, onEnd) {
  const es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    onEvent(ev);
    if (ev.type === "done" || ev.type === "error") { es.close(); onEnd(); }
  };
  es.onerror = () => { es.close(); onEnd(); };
  return () => es.close();
}
