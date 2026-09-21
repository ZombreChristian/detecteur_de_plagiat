(() => {
  const KEY = "ptba-theme";
  const saved = localStorage.getItem(KEY);
  const initial = saved === "dark" || saved === "light" ? saved : "light";
  document.documentElement.dataset.theme = initial;

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
    const label = document.getElementById("ptba-theme-label");
    if (label) label.textContent = theme === "dark" ? "Mode sombre" : "Mode clair";
  }

  function init() {
    if (document.getElementById("ptba-settings")) return;
    const wrap = document.createElement("div");
    wrap.id = "ptba-settings";
    wrap.className = "theme-settings";
    wrap.innerHTML =
      '<button class="theme-settings-button" type="button" aria-expanded="false" aria-controls="ptba-theme-panel">⚙ <span>Paramètres</span></button>' +
      '<div id="ptba-theme-panel" class="theme-settings-panel" hidden>' +
      '<strong>Apparence</strong>' +
      '<button type="button" data-theme-choice="light">☀ Mode clair</button>' +
      '<button type="button" data-theme-choice="dark">☾ Mode sombre</button>' +
      '<div id="ptba-theme-label" class="theme-current"></div>' +
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
    apply(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();