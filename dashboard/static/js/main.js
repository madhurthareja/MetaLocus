import { on, loadSessionList, loadSession, loadStaticData } from './state.js';
import { icon, refreshIcons } from './icons.js';
import * as Overview from './pages/overview.js';
import * as WorldMemory from './pages/worldmemory.js';
import * as HazardTimeline from './pages/hazardtimeline.js';
import * as ExperimentLab from './pages/experimentlab.js';
import * as System from './pages/system.js';

const PAGES = [
  { id: 'overview', label: 'Overview', icon: 'layout-dashboard', mod: Overview },
  { id: 'world-memory', label: 'World Memory', icon: 'globe', mod: WorldMemory },
  { id: 'hazard-timeline', label: 'Hazard Timeline', icon: 'history', mod: HazardTimeline },
  { id: 'experiment-lab', label: 'Experiment Lab', icon: 'flask-conical', mod: ExperimentLab },
  { id: 'system', label: 'System', icon: 'cpu', mod: System },
];

function buildShell() {
  document.getElementById('app').innerHTML = `
    <div class="app-shell">
      <aside class="app-sidebar">
        <div class="brand">
          <div class="brand-mark"><svg viewBox="0 0 24 24" fill="none"><path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" stroke="#fff" stroke-width="1.6"/><circle cx="12" cy="12" r="3" fill="#fff"/></svg></div>
          <div class="brand-word">Locus</div>
        </div>
        <ul class="nav-list" id="nav-list">
          ${PAGES.map((p, i) => `<li><button class="nav-item ${i === 0 ? 'active' : ''}" data-page="${p.id}">${icon(p.icon)}<span>${p.label}</span></button></li>`).join('')}
        </ul>
        <div class="sidebar-spacer"></div>
        <div class="hw-card">
          <div class="hw-card-title">Hardware</div>
          <div class="hw-row"><span class="k">${icon('glasses')}Ray-Ban Display</span><span class="v" id="hw-glasses">simulated</span></div>
          <div class="hw-row"><span class="k">${icon('video')}Camera stream</span><span class="v" id="hw-camera">replay</span></div>
          <div class="hw-row"><span class="k">${icon('monitor')}Display res.</span><span class="v">600&times;600</span></div>
          <div class="hw-row"><span class="k">${icon('volume-2')}Audio output</span><span class="v" id="hw-audio">--</span></div>
          <div class="hw-row"><span class="k">${icon('database')}Dataset</span><span class="v" id="hw-dataset">--</span></div>
        </div>
      </aside>

      <header class="app-header">
        <div class="hdr-group"><span class="hw-pill"><span class="dot off" id="hdr-dot"></span><span>Meta Ray-Ban Display &middot; <span id="hdr-hw-status">not connected</span></span></span></div>
        <div class="hdr-divider"></div>
        <div class="hdr-group">
          <span class="hdr-label">Dataset</span>
          <select class="hdr-select" id="hdr-session"></select>
        </div>
        <div class="hdr-divider"></div>
        <div class="hdr-group"><span class="hdr-label">Session</span><span class="badge info" id="hdr-session-status">loading</span></div>
        <div class="hdr-divider"></div>
        <div class="hdr-group"><span class="hdr-label">Elapsed</span><span class="hdr-value mono" id="hdr-elapsed">0.0s</span></div>
        <div class="spacer"></div>
        <div class="profile-chip">L</div>
      </header>

      <main class="app-main">
        ${PAGES.map((p, i) => `<section class="page ${i === 0 ? 'active' : ''}" id="page-${p.id}"></section>`).join('')}
      </main>
    </div>`;
  refreshIcons();
}

function wireNav() {
  const rendered = new Set();
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      const pageId = btn.dataset.page;
      document.querySelectorAll('.nav-item').forEach((b) => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.page').forEach((p) => p.classList.toggle('active', p.id === `page-${pageId}`));
      if (!rendered.has(pageId)) {
        rendered.add(pageId);
        const page = PAGES.find((p) => p.id === pageId);
        page.mod.renderShell(document.getElementById(`page-${pageId}`));
      }
      window.dispatchEvent(new Event('resize')); // let Three.js canvases pick up their now-visible size
    });
  });
  // Render Overview immediately -- it's the landing page and must paint on load.
  rendered.add('overview');
  Overview.renderShell(document.getElementById('page-overview'));
}

function wireHeader() {
  const sel = document.getElementById('hdr-session');
  sel.addEventListener('change', () => loadSession(sel.value));

  on('sessions-loaded', (sessions) => {
    sel.innerHTML = sessions.map((s) => `<option value="${s}">${s}</option>`).join('');
  });
  on('session-loading', ({ name }) => {
    document.getElementById('hdr-session-status').textContent = 'loading';
    document.getElementById('hdr-session-status').className = 'badge info';
  });
  on('session-loaded', ({ name, manifest }) => {
    sel.value = name;
    const isReal = manifest.dataset !== 'synthetic';
    document.getElementById('hdr-session-status').textContent = isReal ? 'real data replay' : 'synthetic demo';
    document.getElementById('hdr-session-status').className = `badge ${isReal ? 'safe' : 'demo'}`;
    document.getElementById('hw-dataset').textContent = `${manifest.dataset} · ${manifest.n_frames}f`;
    document.getElementById('hw-camera').textContent = isReal ? 'real sensor, replay' : 'synthetic, replay';
    document.getElementById('hw-audio').textContent = isReal ? 'SeamlessM4T v2' : 'muted (demo)';
  });
  on('frame-changed', ({ frame }) => {
    if (frame) document.getElementById('hdr-elapsed').textContent = `${frame.t.toFixed(1)}s`;
  });
}

async function boot() {
  buildShell();
  wireNav();
  wireHeader();
  refreshIcons();

  loadStaticData();

  const sessions = await loadSessionList();
  if (!sessions.length) {
    document.getElementById('hdr-session-status').textContent = 'no sessions found';
    return;
  }
  const preferred = sessions.find((s) => s.startsWith('sanpo_')) || sessions[0];
  await loadSession(preferred);
}

boot();
