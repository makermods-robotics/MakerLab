/* MakerLab prototype B — shared shadcn-style chrome.
   Pages declare <body data-stage="collect|train|market" data-live> and include this script. */

(function () {
  const stage = document.body.dataset.stage || '';
  const live = document.body.hasAttribute('data-live');
  if (live) document.body.classList.add('live');

  const topbar = `
  <header class="topbar">
    <a class="brand" href="home-rig-picker.html">
      <img src="/frontend/public/makermods/logo-mark-white.png" alt="">
      <span>MakerLab</span>
    </a>
    <div class="rig-area">
      <button class="rig-chip" data-rig-menu-toggle aria-haspopup="true">
        <span class="dot"></span> bench-01 <span class="caret">▾</span>
      </button>
      <button class="btn ghost icon sm" data-open-drawer aria-label="Robot settings">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3">
          <circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M12.4 3.6 11 5M5 11l-1.4 1.4"/>
        </svg>
      </button>
      <div class="rig-menu" data-rig-menu>
        <div class="tabs-list">
          <button class="tabs-trigger active" data-tab="single">Single</button>
          <button class="tabs-trigger" data-tab="bimanual">Bimanual</button>
        </div>
        <div class="panel active" data-panel="single">
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>bench-01</b><small>ready · active</small></span><span class="badge secondary">open</span></a>
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>classroom-01</b><small>ready</small></span><span class="badge secondary">open</span></a>
          <a class="rig-row" href="home-rig-picker.html"><span class="dot warn"></span><span><b>travel-arm</b><small>needs calibration</small></span><span class="badge warn">fix</span></a>
        </div>
        <div class="panel" data-panel="bimanual">
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>dual-desk</b><small>ready · 4 arms</small></span><span class="badge secondary">open</span></a>
          <a class="rig-row" href="home-rig-picker.html"><span class="dot warn"></span><span><b>squirrel-01</b><small>right follower port missing</small></span><span class="badge warn">fix</span></a>
        </div>
        <div class="foot"><a href="home-rig-picker.html">+ New robot · manage all</a></div>
      </div>
    </div>
  </header>`;

  const stagebar = `
  <nav class="stagebar" aria-label="Stages">
    <a class="stage-btn ${stage === 'rig' ? 'active' : ''}" href="home-rig-picker.html"><b>Robot</b><small>set up</small></a>
    <a class="stage-btn ${stage === 'collect' ? 'active' : ''}" href="collect.html"><b>Collect</b><small>teach</small></a>
    <a class="stage-btn ${stage === 'train' ? 'active' : ''}" href="train.html"><b>Train &amp; Deploy</b><small>improve</small></a>
    <a class="stage-btn ${stage === 'market' ? 'active' : ''}" href="market.html"><b>Market</b><small>discover</small></a>
  </nav>`;

  const drawer = `
  <div class="scrim" data-close-drawer></div>
  <aside class="drawer" aria-label="Robot settings">
    <div class="drawer-head">
      <div><h2>Robot settings</h2><p class="hint">bench-01 · SO-101 single</p></div>
      <button class="icon-btn" data-close-drawer aria-label="Close">✕</button>
    </div>
    <div class="drawer-body">
      <p class="label">Ports</p>
      <div class="field-row"><span class="field-label">Leader</span><input class="input mono" value="/dev/tty.usbmodem101"><button class="btn outline sm">Detect</button></div>
      <div class="field-row"><span class="field-label">Follower</span><input class="input mono" value="/dev/tty.usbmodem102"><button class="btn outline sm">Detect</button></div>
      <hr class="separator">
      <p class="label">Calibration</p>
      <div class="field-row"><span class="field-label">Leader</span><select class="select"><option>bench-leader-2026-06.json</option><option>+ Calibrate now…</option></select></div>
      <div class="field-row"><span class="field-label">Follower</span><select class="select"><option>bench-follower-2026-06.json</option><option>+ Calibrate now…</option></select></div>
      <hr class="separator">
      <p class="label">Cameras</p>
      <div class="cam-row"><span class="dot"></span> front <span class="meta">1280×720 · 30 fps</span></div>
      <div class="cam-row"><span class="dot"></span> wrist <span class="meta">640×480 · 30 fps</span></div>
      <button class="btn outline sm" style="width:100%">+ Add camera</button>
      <hr class="separator">
      <p class="label">Motor power</p>
      <div class="power">
        <input type="range" min="10" max="100" value="80" oninput="this.nextElementSibling.value=this.value+'%'">
        <output>80%</output>
      </div>
    </div>
    <div class="drawer-foot">
      <button class="btn outline" data-close-drawer>Save</button>
      <a class="btn" href="collect.html">Open robot →</a>
    </div>
  </aside>`;

  const ROBOTS = [
    { name: 'bench-01', mode: 'single', status: 'ready', ok: true, active: true },
    { name: 'dual-desk', mode: 'bimanual', status: 'ready', ok: true },
    { name: 'travel-arm', mode: 'single', status: 'needs calibration', ok: false },
    { name: 'classroom-01', mode: 'single', status: 'ready', ok: true },
    { name: 'squirrel-01', mode: 'bimanual', status: 'port missing', ok: false },
  ];
  const GEAR_SVG = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3"><circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M12.4 3.6 11 5M5 11l-1.4 1.4"/></svg>';
  const sidebar = `
  <aside class="app-side" aria-label="Robots">
    <a class="side-brand" href="home-rig-picker.html">
      <img src="/frontend/public/makermods/logo-mark-white.png" alt="">
      <span>MakerLab</span>
    </a>
    <div class="side-label">Robots</div>
    ${ROBOTS.map((r) => `
    <button class="side-row ${r.active ? 'active' : ''}" data-side-pick="${r.name}">
      <span class="dot ${r.ok ? '' : 'warn'}"></span>
      <span><b>${r.name}</b><small>${r.mode} · ${r.status}</small></span>
      <span class="gear" data-open-drawer aria-label="Configure ${r.name}">${GEAR_SVG}</span>
    </button>`).join('')}
    <div class="side-foot"><button class="btn outline sm" data-open-drawer>+ New robot</button></div>
  </aside>`;

  const selfSide = document.body.hasAttribute('data-self-side');
  document.body.insertAdjacentHTML('afterbegin', topbar);
  if (!live) document.body.insertAdjacentHTML('beforeend', stagebar);
  if (!live && !selfSide) {
    document.body.classList.add('has-sidebar');
    document.body.insertAdjacentHTML('afterbegin', sidebar);
  }
  document.body.insertAdjacentHTML('beforeend', drawer);

  document.addEventListener('click', (e) => {
    const menu = document.querySelector('[data-rig-menu]');
    if (e.target.closest('[data-rig-menu-toggle]')) menu.classList.toggle('open');
    else if (!e.target.closest('[data-rig-menu]')) menu.classList.remove('open');
    const tab = e.target.closest('.rig-menu .tabs-trigger');
    if (tab) {
      menu.querySelectorAll('.tabs-trigger').forEach((t) => t.classList.toggle('active', t === tab));
      menu.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === tab.dataset.tab));
    }
    const sidePick = e.target.closest('[data-side-pick]');
    if (sidePick && !e.target.closest('[data-open-drawer]')) {
      document.querySelectorAll('.app-side .side-row').forEach((r) => r.classList.toggle('active', r === sidePick));
    }
    if (e.target.closest('[data-open-drawer]')) document.body.classList.add('drawer-open');
    if (e.target.closest('[data-close-drawer]')) document.body.classList.remove('drawer-open');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.body.classList.remove('drawer-open');
      document.querySelector('[data-rig-menu]').classList.remove('open');
    }
  });
})();
