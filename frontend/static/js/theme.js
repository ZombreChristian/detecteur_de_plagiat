(() => {
  const KEY = "docsec-theme";
  const saved = localStorage.getItem(KEY);
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
    const label = document.getElementById("docsec-theme-label");
    if (label) label.textContent = theme === "dark" ? "Mode sombre" : "Mode clair";
  }

  function init() {
    if (document.getElementById("docsec-settings")) return;
    const wrap = document.createElement("div");
    wrap.id = "docsec-settings";
    wrap.className = "theme-settings";
    wrap.innerHTML = '<button class="theme-settings-button" type="button" aria-expanded="false" aria-controls="docsec-theme-panel">⚙ <span>Paramètres</span></button>' +
      '<div id="docsec-theme-panel" class="theme-settings-panel" hidden>' +
      '<strong>Apparence</strong>' +
      '<button type="button" data-theme-choice="light">☀ Mode clair</button>' +
      '<button type="button" data-theme-choice="dark">☾ Mode sombre</button>' +
      '<div id="docsec-theme-label" class="theme-current"></div>' +
      '</div>';
    document.body.appendChild(wrap);
    const button = wrap.querySelector(".theme-settings-button");
    const panel = wrap.querySelector(".theme-settings-panel");
    button.addEventListener("click", () => {
      const open = !panel.hidden;
      panel.hidden = open;
      button.setAttribute("aria-expanded", String(!open));
    });
    wrap.querySelectorAll("[data-theme-choice]").forEach(btn => {
      btn.addEventListener("click", () => {
        apply(btn.dataset.themeChoice);
        panel.hidden = true;
        button.setAttribute("aria-expanded", "false");
      });
    });
    apply(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();