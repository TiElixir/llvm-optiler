// Thinking orbs (https://github.com/RareFormLabs/thinking-orbs, MIT): a small canvas animation used
// while a compile is running. Vanilla wrapper around the vendored engine; no framework.
import { MODE_DRAWS } from "../vendor/thinking-orbs/engine/registry.js";
import { resolvePreset } from "../vendor/thinking-orbs/presets.js";

const LABELS = {
  working: "Working…", searching: "Searching…", solving: "Solving…",
  listening: "Listening…", composing: "Composing…", shaping: "Shaping…",
};

/** Each pipeline phase has its own orb, so you can tell where a compile is without reading. */
export function orbStateForPhase(phase) {
  switch (phase) {
    case "queued": case "capture": return "listening";   // reading the model
    case "import": return "composing";                   // building the torch module
    case "passes": return "shaping";                     // MLIR passes reshape the IR
    case "build": return "solving";                      // opt / llc / link
    case "provenance": case "analyze": return "searching"; // tracing lines through the stages
    case "benchmark": return "working";                  // timing the result
    default: return "working";
  }
}

/**
 * Draw an orb into `canvas`. Returns { set(state), destroy() }. Pauses when offscreen or when the tab
 * is hidden, and draws one static frame under prefers-reduced-motion. The UI is dark, so ink is light.
 */
export function mountOrb(canvas, { state = "working", size = 64 } = {}) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  canvas.setAttribute("role", "img");
  const ctx = canvas.getContext("2d");
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  let preset, raf = 0, running = false, visible = true, destroyed = false;
  const frame = () => {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const t = reduced ? 0.6 : (performance.now() / 1000) * preset.speed;
    MODE_DRAWS[preset.mode](ctx, size, t, true, preset.opts);
  };
  const loop = () => { frame(); if (running) raf = requestAnimationFrame(loop); };
  const start = () => { if (!running && !reduced && !destroyed) { running = true; raf = requestAnimationFrame(loop); } };
  const stop = () => { running = false; cancelAnimationFrame(raf); };

  const set = (next) => {
    state = next;
    preset = resolvePreset(state, size);
    canvas.setAttribute("aria-label", LABELS[state] ?? "Working…");
    frame();
  };
  set(state);

  const io = typeof IntersectionObserver !== "undefined" ? new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible && document.visibilityState !== "hidden") start(); else stop();
  }) : null;
  io?.observe(canvas);
  const onVis = () => { if (document.visibilityState === "hidden") stop(); else if (visible) start(); };
  document.addEventListener("visibilitychange", onVis);
  if (!io) start();

  return {
    set,
    destroy() { destroyed = true; stop(); io?.disconnect(); document.removeEventListener("visibilitychange", onVis); },
  };
}
