// tooltip.js — floating replacement for .info-badge's old CSS-only
// `::after` tooltip (project feedback: "справки в панелях не выходят за
// границы полей и текст обрезается"). Root cause: #panel/#left-panel need
// overflow-x:hidden for unrelated reasons (style.css — nowrap content wider
// than the panel used to silently clip instead of wrapping), and that same
// overflow ALSO clipped the `position:absolute` tooltip whenever a badge sat
// near the panel's edge (routine — the panel is narrow, the tooltip is
// 220px). A single `position:fixed` element positioned via
// getBoundingClientRect() on hover isn't confined by an ancestor's
// overflow:hidden the way an absolutely-positioned one is, so one shared
// tooltip node appended near <body> (see index.html:#global-tooltip) sidesteps
// the whole problem — same delegated-listener shape as spin_input.js's
// runtime enhancement of native inputs, for the same reason: badges exist
// both as static markup and inside dynamically-rendered panels (potential's
// accordions, band_lambda's calibration rows, …), so binding once on
// `document` covers all of them without per-badge wiring.

const MARGIN = 8; // px gap kept from the viewport edge when clamping

function _position(el, badge) {
  const r = badge.getBoundingClientRect();
  el.style.left = '0px';
  el.style.top = '0px';
  el.style.display = 'block';
  const w = el.offsetWidth, h = el.offsetHeight;

  let left = r.left + r.width / 2 - w / 2;
  left = Math.min(Math.max(left, MARGIN), window.innerWidth - w - MARGIN);

  // Prefer below the badge (matches the old ::after placement); flip above
  // if there isn't room, so a badge near the bottom of a scrolled panel
  // doesn't push its tooltip off-screen.
  let top = r.bottom + 6;
  if (top + h > window.innerHeight - MARGIN) top = r.top - h - 6;

  el.style.left = `${left}px`;
  el.style.top = `${Math.max(top, MARGIN)}px`;
}

export function initTooltips() {
  const el = document.getElementById('global-tooltip');
  if (!el) return;

  const show = badge => {
    const text = badge.getAttribute('data-tip');
    if (!text) return;
    el.textContent = text;
    _position(el, badge);
  };
  const hide = () => { el.style.display = 'none'; };

  document.addEventListener('mouseover', e => {
    const badge = e.target.closest?.('.info-badge');
    if (badge) show(badge);
  });
  document.addEventListener('mouseout', e => {
    const badge = e.target.closest?.('.info-badge');
    if (badge && !badge.contains(e.relatedTarget)) hide();
  });
  document.addEventListener('focusin', e => {
    const badge = e.target.closest?.('.info-badge');
    if (badge) show(badge);
  });
  document.addEventListener('focusout', e => {
    const badge = e.target.closest?.('.info-badge');
    if (badge) hide();
  });
  // Keep the tooltip glued to its badge during scroll/resize rather than
  // just hiding it — panels scroll independently (#panel/#left-panel) and a
  // stale-positioned tooltip reads as more broken than a live-repositioned
  // one. Cheap: only runs while a tooltip is actually visible.
  const reposition = () => {
    if (el.style.display !== 'block') return;
    const hovered = document.querySelector('.info-badge:hover');
    if (hovered) _position(el, hovered); else hide();
  };
  window.addEventListener('scroll', reposition, true);
  window.addEventListener('resize', reposition);
}
