// Shared by every page, loaded in <head> so the theme is set before anything paints: light or dark from the
// saved choice, else the system's; the button in the tab bar flips it and remembers the choice.  Also shows the
// Power tab once its ratings are ready.
(function () {
  var root = document.documentElement, saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  var system = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function apply(theme) {
    root.setAttribute("data-theme", theme);
    var button = document.querySelector("nav.site .theme");
    if (button) button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  }
  apply(saved || (system && system.matches ? "dark" : "light"));
  if (system && system.addEventListener)
    system.addEventListener("change", function (e) { if (!saved) apply(e.matches ? "dark" : "light"); });
  document.addEventListener("DOMContentLoaded", function () {
    // the Power tab waits until every team has played the games its ratings use (power.json's meta.ready)
    var power = document.querySelector('nav.site a[href="power.html"][hidden]');
    if (power && window.fetch)
      fetch("data/power.json").then(function (r) { return r.json(); })
        .then(function (d) { if (d.meta.ready) power.hidden = false; }).catch(function () {});
    var button = document.querySelector("nav.site .theme");
    if (!button) return;
    apply(root.getAttribute("data-theme"));
    button.addEventListener("click", function () {
      saved = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      try { localStorage.setItem("theme", saved); } catch (e) {}
      apply(saved);
    });
  });
})();
