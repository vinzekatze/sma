// Icon sprite helper — every `<symbol id="icon-*">` lives once in
// index.html's `#icon-sprite` block; this just builds the `<use>` markup for
// template-string contexts (tickers.js, tasks.js, forecast.js,
// forecast_history.js, ...) that render HTML via innerHTML. Static buttons
// written directly in index.html reference the sprite inline instead of
// through this helper — same sprite either way, just two call sites for two
// kinds of markup (server-rendered HTML vs. client-built HTML strings).
//
// extraClass lets a caller add a modifier (e.g. 'icon-fill' for a filled
// star) without a second symbol — see .icon-fill in style.css.
export function iconHtml(name, extraClass = '') {
  const cls = extraClass ? `icon ${extraClass}` : 'icon';
  return `<svg class="${cls}"><use href="#icon-${name}"/></svg>`;
}
