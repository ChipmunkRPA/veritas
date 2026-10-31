// Apply the persisted theme before first paint to avoid a flash. Light is the default.
// A classic blocking script from the same origin, so the PL-04 policy `default-src 'self'` needs no inline-script exception.
(function () {
  try {
    var t = localStorage.getItem('veritas_theme');
    if (t !== 'dark' && t !== 'light') t = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
  } catch (e) { document.documentElement.setAttribute('data-theme', 'light'); }
})();
