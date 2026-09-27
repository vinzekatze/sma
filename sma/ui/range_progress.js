// range_progress.js — WebKit has no equivalent of Firefox's native
// ::-moz-range-progress (a "filled" portion of the track up to the thumb —
// style.css already styles that pseudo-element), so Chrome/Safari show an
// unfilled track with just the handle (project feedback: "у ползунков в
// фоксе есть синяя подсветка, в хроме нет"). Faked here with a CSS custom
// property (--range-pct) driving a linear-gradient background on
// ::-webkit-slider-runnable-track (style.css) — this module's only job is
// keeping --range-pct in sync with the actual value.
//
// A delegated 'input' listener covers every live drag, but a LOT of call
// sites across the app set a slider's .value programmatically when applying
// saved/loaded settings (analysis.js, forecast.js, simplex_ensemble.js,
// trend_ruler.js, variance_oscillator.js, regime_mixture_potential.js,
// risk_corridor.js, …) — a plain property assignment fires no 'input' event
// at all. Chasing down and editing every one of those (present, and every
// future one) isn't worth the ongoing coupling, so instead the `value`
// accessor itself is wrapped once per slider at init — same "one runtime
// pass covers it everywhere, forever" idea spin_input.js already uses for
// number-input spinners, just via a property override instead of a
// MutationObserver (no range slider in this app is created dynamically
// after load, unlike those number inputs, so a single sweep is enough).
function _wrapValueSetter(el) {
  const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
  Object.defineProperty(el, 'value', {
    get() { return desc.get.call(this); },
    set(v) { desc.set.call(this, v); syncRangeProgress(this); },
    configurable: true,
  });
}

export function syncRangeProgress(el) {
  const min = +el.min || 0, max = +el.max || 100, val = +el.value;
  let pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  // style.css's gradient always fills from the LEFT (a fixed "to right"
  // direction) up to --range-pct. A `direction:rtl` slider (potential's
  // rewind slider — deliberately mirrored so it reads left-to-right in
  // time, see index.html) renders its native thumb/fill from the RIGHT
  // instead, so the fraction has to be inverted to still visually match.
  if (getComputedStyle(el).direction === 'rtl') pct = 100 - pct;
  el.style.setProperty('--range-pct', `${pct}%`);
}

export function initRangeProgress() {
  document.querySelectorAll('input[type="range"]').forEach(el => {
    _wrapValueSetter(el);
    syncRangeProgress(el);
  });
  // Redundant with _wrapValueSetter for a plain user drag (the browser sets
  // .value internally too, which the wrapped setter already catches) —
  // kept anyway as a harmless belt-and-suspenders in case some interaction
  // path ever sets the underlying attribute instead of the property.
  document.addEventListener('input', e => {
    if (e.target.matches?.('input[type="range"]')) syncRangeProgress(e.target);
  });
}
