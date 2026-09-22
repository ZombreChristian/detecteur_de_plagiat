(() => {
const KEY="docsec-theme";
const SIDEBAR_KEY="tdrdoc-sidebar";
const saved=localStorage.getItem(KEY);
const savedSidebar=localStorage.getItem(SIDEBAR_KEY);

if(saved==="light"||saved==="dark") document.documentElement.dataset.theme=saved;
document.documentElement.dataset.sidebar=(savedSidebar==="collapsed")?"collapsed":"expanded";

function apply(theme){
  document.documentElement.dataset.theme=theme;
  localStorage.setItem(KEY,theme);
  const l=document.getElementById("docsec-theme-label");
  if(l) l.textContent=theme==="dark"?"Mode sombre":"Mode clair";
}

function applySidebar(state){
  document.documentElement.dataset.sidebar=state;
  localStorage.setItem(SIDEBAR_KEY,state);
  const collapsed=state==="collapsed";
  document.querySelectorAll(".rail-collapse,.tdr-topbar-toggle").forEach(btn=>{
    btn.setAttribute("aria-expanded",String(!collapsed));
    btn.setAttribute("aria-label",collapsed?"Déployer la barre latérale":"Réduire la barre latérale");
    btn.setAttribute("title",collapsed?"Déployer la barre latérale":"Réduire la barre latérale");
    const icon=btn.querySelector("svg");
    if(icon){
      icon.innerHTML=collapsed
        ? '<path d="M8 5l7 7-7 7"></path>'
        : '<path d="M16 5l-7 7 7 7"></path>';
    }
  });
}

function initSidebar(){
  const rail=document.querySelector(".rail");
  const nav=document.querySelector(".rail-nav");
  const main=document.querySelector(".main");
  if(!rail||!nav||!main)return;

  if(!document.getElementById("rail-collapse")){
    const button=document.createElement("button");
    button.id="rail-collapse";
    button.className="rail-collapse";
    button.type="button";
    button.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 5l-7 7 7 7"></path></svg>';
    rail.appendChild(button);
    button.addEventListener("click",()=>{
      applySidebar(document.documentElement.dataset.sidebar==="collapsed"?"expanded":"collapsed");
    });
  }
  applySidebar(document.documentElement.dataset.sidebar==="collapsed"?"collapsed":"expanded");
}

function initTopbar(){
  const main=document.querySelector(".main");
  if(!main || document.getElementById("tdr-topbar"))return;

  const ctxTitle=main.querySelector(".ctxbar h1");
  const pageTitle=ctxTitle ? ctxTitle.textContent.trim() : "Espace de travail";
  const userNode=document.querySelector(".rail-user strong");
  const user=userNode ? userNode.textContent.trim() : "";

  const bar=document.createElement("div");
  bar.id="tdr-topbar";
  bar.className="tdr-topbar";

  const left=document.createElement("div");
  left.className="tdr-topbar-left";
  const toggle=document.createElement("button");
  toggle.className="tdr-topbar-toggle";
  toggle.type="button";
  toggle.setAttribute("aria-expanded","true");
  toggle.setAttribute("aria-label","Réduire la barre latérale");
  toggle.setAttribute("title","Réduire la barre latérale");
  toggle.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 5l-7 7 7 7"></path></svg>';
  const brand=document.createElement("span");
  brand.className="tdr-topbar-brand";
  brand.textContent="TDRDOC-SCAN";
  const separator=document.createElement("span");
  separator.className="tdr-topbar-separator";
  const page=document.createElement("span");
  page.className="tdr-topbar-page";
  page.textContent=pageTitle;

  left.append(toggle,brand,separator,page);

  const right=document.createElement("div");
  right.className="tdr-topbar-right";
  const accent=document.createElement("span");
  accent.className="tdr-topbar-accent";
  accent.setAttribute("aria-hidden","true");
  const status=document.createElement("span");
  status.className="tdr-topbar-status";
  status.textContent="Session active";
  const userEl=document.createElement("span");
  userEl.className="tdr-topbar-user";
  userEl.textContent=user;

  right.append(accent,status,userEl);
  bar.append(left,right);
  main.insertBefore(bar,main.firstChild);

  toggle.addEventListener("click",()=>{
    applySidebar(document.documentElement.dataset.sidebar==="collapsed"?"expanded":"collapsed");
  });
}

function initThemeSettings(){
  if(document.getElementById("docsec-settings"))return;
  const nav=document.querySelector(".rail-nav");
  if(!nav)return;

  const w=document.createElement("div");
  w.id="docsec-settings";
  w.className="rail-settings";
  w.innerHTML='<button class="rail-link rail-settings-trigger" type="button" aria-expanded="false" aria-controls="docsec-theme-panel"><span class="ico"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"></path><circle cx="12" cy="12" r="4"></circle></svg></span><span>Paramètres</span></button><div id="docsec-theme-panel" class="theme-settings-panel" hidden><strong>Apparence</strong><button type="button" data-theme-choice="light">Mode clair</button><button type="button" data-theme-choice="dark">Mode sombre</button><div id="docsec-theme-label" class="theme-current"></div></div>';
  nav.appendChild(w);

  const b=w.querySelector(".rail-settings-trigger");
  const p=w.querySelector(".theme-settings-panel");
  b.addEventListener("click",()=>{
    const open=!p.hidden;
    p.hidden=open;
    b.setAttribute("aria-expanded",String(!open));
  });

  w.querySelectorAll("[data-theme-choice]").forEach(x=>x.addEventListener("click",()=>{
    apply(x.dataset.themeChoice);
    p.hidden=true;
    b.setAttribute("aria-expanded","false");
  }));

  apply(document.documentElement.dataset.theme==="light"?"light":"dark");
}

function init(){
  initSidebar();
  initTopbar();
  initThemeSettings();
}

if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",init);
else init();
})();