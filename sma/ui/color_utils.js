// color_utils.js — tiny shared helpers for turning a color-profile hex value
// (docs/plans/frontend_improvements_plan.md §1.1a) into the string forms
// Plotly actually wants: an "r,g,b" triplet (for building an rgba(...) fill
// with a caller-chosen alpha) or a ready-made rgba(...) string. Several
// tools need a SEMI-transparent version of a configurable color (band zones,
// simplex origin lines) — the alpha itself stays a fixed
// per-feature constant (not user-configurable, keeps the color picker to
// one control per role instead of a color+alpha pair everywhere), only the
// hue comes from the profile.

export function hexToRgbTriplet(hex) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `${r},${g},${b}`;
}

export function hexToRgba(hex, alpha) {
  return `rgba(${hexToRgbTriplet(hex)},${alpha})`;
}
