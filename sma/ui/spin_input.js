// spin_input.js — custom cross-browser replacement for native
// input[type=number] spin buttons (project feedback 2026-08-25: "спинеры...
// по-прежнему не кастомизированы и по-разному отображаются в хроме и
// файрфоксе"). Native spin buttons can't be restyled via CSS (only hidden —
// there is no property that recolors them), so the only way to make them
// look the same in both browsers is to hide the native ones and draw our
// own (style.css hides them project-wide now; this module supplies the
// replacement).
//
// Wraps every input[type=number] at RUNTIME (MutationObserver), not by
// editing markup — ~20 number inputs live as static HTML (index.html), but
// several more are generated dynamically by template strings across
// forecast.js (calibration target rows), moving_averages.js, trend_ruler.js,
// zigzag_tool.js — editing all of those to add wrapper markup by hand would
// mean touching 5+ files and keeping them in sync forever after; a single
// runtime pass covers all of them, present and future, with zero coupling.
//
// .no-spin (an existing class — #tr-window/#vo-window, each paired with an
// adjacent range slider, project feedback 2026-08-19: spinner is genuinely
// redundant there) opts a field OUT of wrapping entirely, same class CSS
// already uses to hide native buttons on those two.

function stepInput(input, dir) {
  if (dir > 0) input.stepUp(); else input.stepDown();
  // stepUp()/stepDown() change .value WITHOUT firing input/change — every
  // existing onchange/oninput handler on these fields (most of them have
  // one) needs a synthetic event to actually react to the new value.
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

function wrapSpinInput(input) {
  if (input.classList.contains('no-spin')) return;
  if (input.closest('.spin-input-wrap')) return; // already wrapped (incl. re-visits from the same DOM mutation)

  const wrap = document.createElement('span');
  wrap.className = 'spin-input-wrap';
  // The wrapper takes over any layout-affecting inline style (width, flex,
  // ...) the input had — it's now the box-model participant in whatever row
  // this input sits in; the input itself just fills it (style.css:
  // .spin-input-wrap input[type="number"] { width: 100% }).
  if (input.style.cssText) {
    wrap.style.cssText = input.style.cssText;
    input.style.cssText = '';
  }
  input.parentNode.insertBefore(wrap, input);
  wrap.appendChild(input);

  const btns = document.createElement('span');
  btns.className = 'spin-btns';
  btns.innerHTML = `
    <button type="button" class="spin-btn spin-up" tabindex="-1">▲</button>
    <button type="button" class="spin-btn spin-down" tabindex="-1">▼</button>
  `;
  wrap.appendChild(btns);

  btns.querySelector('.spin-up').addEventListener('click', () => stepInput(input, 1));
  btns.querySelector('.spin-down').addEventListener('click', () => stepInput(input, -1));
}

function wrapAllIn(root) {
  if (root.matches?.('input[type="number"]')) wrapSpinInput(root);
  root.querySelectorAll?.('input[type="number"]').forEach(wrapSpinInput);
}

export function initSpinInputs() {
  wrapAllIn(document.body);
  new MutationObserver(mutations => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType === 1) wrapAllIn(node);
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
}
