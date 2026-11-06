// Number formatting. Every figure the UI shows goes through here so units stay consistent.

export function fmtMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "–";
  if (ms >= 100) return ms.toFixed(0);
  if (ms >= 10) return ms.toFixed(1);
  if (ms >= 1) return ms.toFixed(2);
  return ms.toFixed(3);
}

export function fmtRatio(r) {
  if (r == null || !Number.isFinite(r)) return "–";
  return r >= 10 ? r.toFixed(1) : r.toFixed(2);
}

export function fmtBytes(n) {
  if (n == null) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / 1024 / 1024).toFixed(2)} MiB`;
}

export function fmtPct(p) {
  return p == null ? "–" : `${Math.round(p * 100)}%`;
}

export const pad2 = (n) => String(n).padStart(2, "0");
export const pad3 = (n) => String(n).padStart(3, "0");

export function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
