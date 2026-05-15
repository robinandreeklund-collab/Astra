// Astra frontend bootstrap
(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  ready(function () {
    if (typeof window.__astra_init === 'function') {
      try { window.__astra_init(); } catch (e) { console.error(e); }
    }
  });
})();
