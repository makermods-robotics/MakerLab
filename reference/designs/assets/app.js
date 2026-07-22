/* MakerLab prototype — shared app chrome.
   Pages declare <body data-stage="collect|train|market" data-live> and include this script.
   Injects: topbar (logo → home, rig chip + dropdown, gear), stage bar, rig settings drawer. */

(function () {
  const stage = document.body.dataset.stage || '';
  const live = document.body.hasAttribute('data-live');
  if (live) document.body.classList.add('live');

  const topbar = `
  <header class="topbar">
    <a class="brand" href="home-rig-picker.html">
      <img src="/frontend/public/makermods/logo-mark-white.png" alt="">
      <span>MAKERLAB</span>
    </a>
    <div class="rig-area">
      <button class="rig-chip" data-rig-menu-toggle aria-haspopup="true">
        <span class="dot"></span> bench-rig <span class="caret">▾</span>
      </button>
      <button class="gear-btn" data-open-drawer aria-label="Rig settings">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3">
          <circle cx="8" cy="8" r="2.2"/><path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M12.4 3.6 11 5M5 11l-1.4 1.4"/>
        </svg>
      </button>
      <div class="rig-menu" data-rig-menu>
        <div class="tabs">
          <button class="tab active" data-tab="single">Single</button>
          <button class="tab" data-tab="bimanual">Bimanual</button>
        </div>
        <div class="panel active" data-panel="single">
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>bench-rig</b><small>ready · active</small></span><span class="tag">open</span></a>
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>classroom-01</b><small>ready</small></span><span class="tag">open</span></a>
          <a class="rig-row" href="home-rig-picker.html"><span class="dot warn"></span><span><b>travel-arm</b><small>needs calibration</small></span><span class="tag">fix</span></a>
        </div>
        <div class="panel" data-panel="bimanual">
          <a class="rig-row" href="collect.html"><span class="dot"></span><span><b>dual-desk</b><small>ready · 4 arms</small></span><span class="tag">open</span></a>
          <a class="rig-row" href="home-rig-picker.html"><span class="dot warn"></span><span><b>squirrel-rig</b><small>right follower port missing</small></span><span class="tag">fix</span></a>
        </div>
        <div class="foot"><a href="home-rig-picker.html">+ New rig · manage all</a></div>
      </div>
    </div>
  </header>`;

  const stagebar = `
  <nav class="stagebar" aria-label="Stages">
    <a class="stage-btn ${stage === 'rig' ? 'active' : ''}" href="home-rig-picker.html"><b><span class="num">①</span>RIG</b><small>set up your robot</small></a>
    <a class="stage-btn ${stage === 'collect' ? 'active' : ''}" href="collect.html"><b><span class="num">②</span>COLLECT</b><small>teach by demonstration</small></a>
    <a class="stage-btn ${stage === 'train' ? 'active' : ''}" href="train.html"><b><span class="num">③</span>TRAIN &amp; DEPLOY</b><small>turn demos into skills</small></a>
    <a class="stage-btn ${stage === 'market' ? 'active' : ''}" href="market.html"><b><span class="num">④</span>MARKET</b><small>get skills &amp; data</small></a>
  </nav>`;

  const drawer = `
  <div class="scrim" data-close-drawer></div>
  <aside class="drawer" aria-label="Rig settings">
    <div class="drawer-head">
      <div><span class="eyebrow">Rig settings</span><h2>bench-rig</h2></div>
      <button class="icon-btn" data-close-drawer aria-label="Close">✕</button>
    </div>
    <div class="drawer-body">
      <div class="field-group">
        <span class="eyebrow">Ports</span>
        <div class="field-row"><span class="field-label">Leader</span><input class="field-input" value="/dev/tty.usbmodem101"><button class="btn">Detect</button></div>
        <div class="field-row"><span class="field-label">Follower</span><input class="field-input" value="/dev/tty.usbmodem102"><button class="btn">Detect</button></div>
      </div>
      <div class="field-group">
        <span class="eyebrow">Calibration</span>
        <div class="field-row"><span class="field-label">Leader</span><select class="field-input"><option>bench-leader-2026-06.json</option><option>+ Calibrate now…</option></select></div>
        <div class="field-row"><span class="field-label">Follower</span><select class="field-input"><option>bench-follower-2026-06.json</option><option>+ Calibrate now…</option></select></div>
      </div>
      <div class="field-group">
        <span class="eyebrow">Cameras</span>
        <div class="cam-row"><span class="dot"></span> front <span class="meta">1280×720 · 30 fps</span></div>
        <div class="cam-row"><span class="dot"></span> wrist <span class="meta">640×480 · 30 fps</span></div>
        <div class="field-row"><button class="btn" style="width:100%">+ Add camera</button></div>
      </div>
      <div class="field-group">
        <span class="eyebrow">Motor power</span>
        <div class="power">
          <input type="range" min="10" max="100" value="80" oninput="this.nextElementSibling.value=this.value+'%'">
          <output>80%</output>
        </div>
      </div>
    </div>
    <div class="drawer-foot">
      <button class="btn" data-close-drawer>Save</button>
      <a class="btn-brand" href="collect.html">Open rig →</a>
    </div>
  </aside>`;

  document.body.insertAdjacentHTML('afterbegin', topbar);
  if (!live) document.body.insertAdjacentHTML('beforeend', stagebar);
  document.body.insertAdjacentHTML('beforeend', drawer);

  document.addEventListener('click', (e) => {
    const menu = document.querySelector('[data-rig-menu]');
    if (e.target.closest('[data-rig-menu-toggle]')) menu.classList.toggle('open');
    else if (!e.target.closest('[data-rig-menu]')) menu.classList.remove('open');
    const tab = e.target.closest('.rig-menu .tab');
    if (tab) {
      menu.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      menu.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === tab.dataset.tab));
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
