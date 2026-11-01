import { state, on, currentFrame, mostUrgentEdge, hazardSeverity, clockHour, nodeById, selectNode, hoverNode, jumpToEvent, allHazardEvents, play, pause, togglePlay, setSpeed, setShowMasks, setIdx } from './state.js';
import { icon, refreshIcons } from './icons.js';

function fmtS(t) { return `${t.toFixed(1)}s`; }
function el(html) { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; }

// A small positioned popover, replacing native alert()/title tooltips for
// "click for detail" affordances. Clicking the same anchor again closes it.
// Exported -- every page uses this same disclosure instead of inventing its
// own, so "compact by default, full detail on click" reads consistently.
export function showPopover(anchorEl, html, key) {
  const existing = document.querySelector('.popover');
  if (existing) {
    const toggledOff = existing.dataset.key === key;
    existing.remove();
    if (toggledOff) return;
  }
  const pop = document.createElement('div');
  pop.className = 'popover';
  pop.dataset.key = key;
  pop.innerHTML = html;
  document.body.appendChild(pop);
  refreshIcons();
  const r = anchorEl.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let left = Math.max(8, Math.min(r.left + r.width / 2 - pw / 2, window.innerWidth - pw - 8));
  let top = r.bottom + 8;
  if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - ph - 8);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
  function onDocClick(ev) {
    if (pop.contains(ev.target)) return;
    pop.remove();
    document.removeEventListener('mousedown', onDocClick, true);
  }
  setTimeout(() => document.addEventListener('mousedown', onDocClick, true), 0);
}

// =====================================================================
// Live Camera Feed  (docs/UI.md section 5.B)
// =====================================================================
export class CameraFeed {
  constructor(container) {
    this.container = container;
    this.container.innerHTML = `
      <div class="card-header">
        <div><h3>Live Camera Feed</h3><div class="sub">Real SAM 3.1 segmentation, replayed frame-by-frame</div></div>
        <div class="btn-group">
          <button class="btn mask-toggle active" data-v="1">Masks</button>
          <button class="btn mask-toggle" data-v="0">Raw</button>
        </div>
      </div>
      <div class="card-body flush">
        <div class="camera-wrap">
          <video id="cf-video-frames" muted playsinline preload="auto"></video>
          <video id="cf-video-masks" muted playsinline preload="auto"></video>
          <div class="camera-overlay-top">
            <div class="live-chip"><span class="reddot"></span><span id="cf-state">REPLAY · PAUSED</span></div>
            <div class="cam-meta-chip mono"><div id="cf-res">640×360</div><div id="cf-t">t=0.0s · frame 0</div></div>
          </div>
          <div class="camera-overlay-bottom" id="cf-chips"></div>
          <div class="cam-toolbar"><button class="icon-btn" id="cf-expand">${icon('maximize-2')}</button></div>
        </div>
      </div>`;
    this.vFrames = this.container.querySelector('#cf-video-frames');
    this.vMasks = this.container.querySelector('#cf-video-masks');

    this.container.querySelectorAll('.mask-toggle').forEach((b) =>
      b.addEventListener('click', () => { setShowMasks(b.dataset.v === '1'); this._syncMaskBtn(); })
    );
    this.container.querySelector('#cf-expand').addEventListener('click', () => {
      const card = this.container.closest('.card');
      const nowFullscreen = card.classList.toggle('fullscreen-card');
      this.container.querySelector('#cf-expand').innerHTML = icon(nowFullscreen ? 'minimize-2' : 'maximize-2');
      refreshIcons();
    });
    on('frame-changed', () => this.render());
    on('session-loaded', ({ name }) => {
      // One request per video instead of one per frame -- the browser's own
      // buffering replaces the old preloadImages() flood, and src is only
      // ever (re)assigned here, not per frame, so playback doesn't reload
      // the video on every tick.
      this.vFrames.src = `/data/${name}/frames.mp4`;
      this.vMasks.src = `/data/${name}/masks.mp4`;
      this.render();
    });
    on('masks-toggled', () => this.render());
    on('playback-changed', () => this.render());
    on('node-hovered', ({ nodeId }) => this._highlightChip(nodeId));
    refreshIcons();
    if (state.manifest) this.render();
  }

  _syncMaskBtn() {
    this.container.querySelectorAll('.mask-toggle').forEach((b) => b.classList.toggle('active', (b.dataset.v === '1') === state.showMasks));
  }

  _highlightChip(nodeId) {
    this.container.querySelectorAll('.cf-chip').forEach((c) => c.classList.toggle('active', Number(c.dataset.id) === nodeId));
  }

  // All-intra encoding (see generate_dashboard_data.py) makes a seek per
  // timeline tick cheap -- each frame is an independent keyframe, so this
  // never has to decode backward through a GOP to land on it. Both videos
  // stay seeked in lockstep (not just the visible one) so toggling
  // Masks/Raw never shows a stale frame.
  render() {
    const frame = currentFrame();
    if (!frame || !state.session) return;
    const active = state.showMasks ? this.vMasks : this.vFrames;
    const inactive = state.showMasks ? this.vFrames : this.vMasks;
    active.style.display = 'block';
    inactive.style.display = 'none';
    [this.vFrames, this.vMasks].forEach((v) => {
      const t = Number.isFinite(v.duration) ? Math.min(frame.t, Math.max(0, v.duration - 1 / 60)) : frame.t;
      if (Math.abs(v.currentTime - t) > 1e-3) v.currentTime = t;
    });
    this.container.querySelector('#cf-t').textContent = `t=${fmtS(frame.t)} · frame ${frame.frame}`;
    this.container.querySelector('#cf-res').textContent = `${state.manifest.dataset} · 640px`;
    const stateChip = this.container.querySelector('#cf-state');
    stateChip.textContent = `REPLAY · ${state.playing ? 'PLAYING' : 'PAUSED'}`;
    this.container.querySelector('.reddot').style.background = state.playing ? '#16865D' : '#9FB4BD';

    const visibleNodes = frame.nodes.filter((n) => n.visible);
    const chips = this.container.querySelector('#cf-chips');
    chips.innerHTML = `<div style="display:flex; gap:6px; flex-wrap:wrap; pointer-events:auto;">${
      visibleNodes.length
        ? visibleNodes.map((n) => `<span class="detect-label-chip cf-chip" style="position:static; transform:none;" data-id="${n.node_id}">#${n.node_id} ${n.concept}</span>`).join('')
        : '<span class="detect-label-chip" style="position:static; transform:none; opacity:0.6;">no detections this frame</span>'
    }</div>`;
    chips.querySelectorAll('.cf-chip').forEach((c) => {
      c.style.cursor = 'pointer';
      c.addEventListener('click', () => selectNode(Number(c.dataset.id)));
    });
    this._highlightChip(state.selectedNodeId);
  }
}

// =====================================================================
// Active Hazard Panel  (docs/UI.md section 5.C)
// =====================================================================
function compassSvg(bearingDeg, color) {
  const cx = 32, cy = 32, r = 24;
  const angleRad = ((bearingDeg - 90) * Math.PI) / 180;
  const x = cx + r * Math.cos(angleRad), y = cy + r * Math.sin(angleRad);
  return `<svg viewBox="0 0 64 64" width="64" height="64">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--border)" stroke-width="2"/>
    <line x1="${cx}" y1="${cy - r}" x2="${cx}" y2="${cy - r + 6}" stroke="var(--text-secondary)" stroke-width="2"/>
    <circle cx="${x}" cy="${y}" r="5" fill="${color}"/>
    <circle cx="${cx}" cy="${cy}" r="2.5" fill="var(--text-secondary)"/>
  </svg>`;
}

export class HazardPanel {
  constructor(container) {
    this.container = container;
    on('frame-changed', () => this.render());
    on('session-loaded', () => this.render());
    this.render();
  }

  render() {
    const frame = currentFrame();
    const card = this.container.closest('.hazard-card') || this.container;
    if (!frame) { this.container.innerHTML = '<div class="inspector-empty">no session loaded</div>'; return; }

    const edge = mostUrgentEdge(frame.edges);
    const sev = hazardSeverity(edge);
    card.classList.remove('state-safe', 'state-advisory', 'state-hazard', 'state-critical');
    card.classList.add(sev === 'critical' ? 'state-critical' : sev === 'hazard' ? 'state-hazard' : sev === 'advisory' ? 'state-advisory' : 'state-safe');

    if (!edge) {
      this.container.innerHTML = `
        <div class="hazard-headline">
          <div class="glyph">${icon('shield-check')}</div>
          <div><div class="hazard-title">No active hazard</div><div class="hazard-subtitle">Nothing within ${state.manifest.edge_radius_m}m is on a collision-relevant course.</div></div>
        </div>
        <div class="notice-box" style="margin-top:14px;">${icon('info')}<span>No hazard detected does not mean the route is safe -- only that nothing inside the current sensing radius meets the warning criteria.</span></div>`;
      refreshIcons();
      return;
    }

    // "advisory" means an edge exists (something is within sensing radius)
    // but ttc_s is null -- i.e. it is NOT on a closing/collision course. The
    // real spoken directive reflects that (directives.py only adds urgency
    // language once ttc_s is a real, closing number), so the dashboard must
    // not say "APPROACHING" or show an alert glyph for a non-closing object
    // -- that would be a false alarm this accessibility tool cannot afford.
    const isClosing = sev === 'hazard' || sev === 'critical';
    const node = nodeById(edge.node_id);
    const hour = clockHour(edge.bearing_deg);
    const side = edge.bearing_deg > 5 ? 'RIGHT' : edge.bearing_deg < -5 ? 'LEFT' : 'AHEAD';
    const color = sev === 'critical' ? 'var(--critical)' : sev === 'hazard' ? 'var(--hazard)' : 'var(--blue)';
    const visibility = node?.visible ? 'Currently in camera frame' : `Out of camera frame · last observed ${(frame.t - (node?.last_seen ?? frame.t)).toFixed(1)}s ago`;
    const speed = node ? Math.hypot(...node.velocity_world) : 0;
    const motionDesc = speed > 0.15 ? `moving at ~${speed.toFixed(1)} m/s` : 'approximately stationary';

    this.container.innerHTML = `
      <div class="hazard-headline">
        <div class="glyph">${icon(sev === 'critical' ? 'octagon-alert' : isClosing ? 'triangle-alert' : 'eye')}</div>
        <div>
          <div class="hazard-title">${isClosing ? `APPROACHING FROM ${side}` : `TRACKED NEARBY · NOT CLOSING`}</div>
          <div class="hazard-subtitle">${edge.concept} · track #${edge.node_id}</div>
        </div>
        ${compassSvg(edge.bearing_deg, color)}
      </div>
      <div class="hazard-directive-line">"${edge.directive}"</div>
      <ul class="hazard-reason-list">
        <li><span class="k">${isClosing ? 'Direction of approach' : 'Direction'}</span><span>${hour} o'clock · ${edge.bearing_deg.toFixed(0)}°</span></li>
        <li><span class="k">Current visibility</span><span>${node?.visible ? 'Observed this frame' : 'Retained in memory'}</span></li>
        <li><span class="k">Last observed state</span><span>${visibility}</span></li>
        <li><span class="k">Estimated trajectory</span><span>${motionDesc}</span></li>
        <li><span class="k">Range</span><span>${edge.range_m.toFixed(1)} m</span></li>
        <li><span class="k">Time-to-collision</span><span class="mono">${edge.ttc_s != null ? edge.ttc_s.toFixed(1) + 's' : 'n/a (not closing)'}</span></li>
        <li><span class="k">Spatial tone</span><span>${isClosing ? `panned ${side.toLowerCase()}, pulse scales with closing TTC` : `panned ${side.toLowerCase()}, no urgency pulse (not closing)`}</span></li>
      </ul>`;
    refreshIcons();
  }
}

// =====================================================================
// Ray-Ban Display HUD preview  (docs/UI.md section 10)
// =====================================================================
function hudArc(hour, sev) {
  const cx = 30, cy = 30, r = 22;
  const angleRad = ((hour / 12) * 360 - 90) * (Math.PI / 180);
  const x = cx + r * Math.cos(angleRad), y = cy + r * Math.sin(angleRad);
  const color = sev === 'critical' ? '#FF6B6B' : sev === 'hazard' ? '#FFB454' : '#3E9BFF';
  return `<svg width="60" height="36" viewBox="0 0 60 44">
    <path d="M 6 30 A 24 24 0 0 1 54 30" fill="none" stroke="#161616" stroke-width="2"/>
    <circle cx="${x}" cy="${y}" r="4" fill="${color}"/>
  </svg>`;
}

export class HudPreview {
  constructor(container) {
    this.container = container;
    container.innerHTML = `
      <div class="hud-outer"><div class="hud-lens" id="hud-lens"><div class="hud-state-label" id="hud-statelabel">PATH CLEAR</div></div></div>
      <div class="hud-caption-row"><span>Spoken directive <span class="spoken" id="hud-spoken">--</span></span><span id="hud-tone">tone: --</span></div>`;
    on('frame-changed', () => this.render());
    on('session-loaded', () => this.render());
    this.render();
  }

  render() {
    const frame = currentFrame();
    const lens = this.container.querySelector('#hud-lens');
    if (!frame) return;
    const edge = mostUrgentEdge(frame.edges);
    const sev = hazardSeverity(edge);

    if (!edge) {
      lens.innerHTML = `<div class="hud-state-label">STATE 1 · NO HAZARD</div><div class="hud-directive-text" style="font-size:17px; color:#4F8A91;">PATH CLEAR</div>`;
      this.container.querySelector('#hud-spoken').textContent = '--';
      this.container.querySelector('#hud-tone').textContent = 'tone: silent';
      return;
    }
    // Same rule as the hazard panel: only an edge with a real, closing
    // ttc_s earns alarm wording -- an advisory (tracked, not closing) edge
    // must not render "CAUTION" on the simulated glasses HUD.
    const isClosing = sev === 'hazard' || sev === 'critical';
    const hour = clockHour(edge.bearing_deg);
    const side = edge.bearing_deg > 5 ? 'RIGHT' : edge.bearing_deg < -5 ? 'LEFT' : 'AHEAD';
    const label = sev === 'critical' ? 'STOP' : isClosing ? `CAUTION ${side}` : `${edge.concept.toUpperCase()} ${side}`;
    const words = edge.directive.split(' ').slice(0, 6).join(' ');
    const stateLabel = sev === 'critical' ? '3 · CRITICAL' : sev === 'hazard' ? '2 · APPROACHING' : '1 · TRACKED';
    lens.innerHTML = `
      <div class="hud-state-label">STATE ${stateLabel}</div>
      <div class="hud-arc-wrap">${hudArc(hour, sev)}</div>
      <div class="hud-directive-text ${isClosing ? sev : ''}">${label}</div>
      <div class="hud-sub">${edge.range_m.toFixed(0)}m${edge.ttc_s != null ? ' · ' + edge.ttc_s.toFixed(1) + 's' : ''}</div>`;
    this.container.querySelector('#hud-spoken').textContent = `"${words}"`;
    const pan = edge.bearing_deg > 5 ? 'right' : edge.bearing_deg < -5 ? 'left' : 'center';
    this.container.querySelector('#hud-tone').textContent = `tone: panned ${pan}`;
  }
}

// =====================================================================
// Event Timeline  (docs/UI.md section 5.D)
// =====================================================================
export class EventTimeline {
  constructor(container) {
    this.container = container;
    this.container.innerHTML = `
      <div class="timeline-controls">
        <button class="icon-btn" id="tl-play">${icon('play')}</button>
        <select class="hdr-select" id="tl-speed">
          <option value="2">2x</option><option value="1" selected>1x</option><option value="0.5">0.5x</option><option value="0.25">0.25x</option>
        </select>
        <span class="timeline-time mono" id="tl-time">0.0s / 0.0s</span>
        <span class="spacer" style="flex:1"></span>
        <span class="hdr-label">Jump to hazard</span>
        <button class="icon-btn" id="tl-prev-haz">${icon('skip-back')}</button>
        <button class="icon-btn" id="tl-next-haz">${icon('skip-forward')}</button>
      </div>
      <div class="timeline-track-wrap">
        <div class="timeline-scrub">
          <div class="timeline-rail"></div>
          <div class="timeline-progress" id="tl-progress"></div>
          <div id="tl-markers"></div>
          <input type="range" id="tl-scrub" min="0" max="0" value="0">
        </div>
        <div class="timeline-legend-row">
          <span class="legend-row"><span class="legend-swatch" style="background:#9FB4BD"></span>Detection</span>
          <span class="legend-row"><span class="legend-swatch" style="background:var(--blue)"></span>Re-association</span>
          <span class="legend-row"><span class="legend-swatch" style="background:var(--hazard)"></span>Hazard warning</span>
          <span class="legend-row"><span class="legend-swatch" style="background:var(--critical)"></span>Critical warning</span>
        </div>
      </div>`;
    this.container.querySelector('#tl-play').addEventListener('click', () => togglePlay());
    this.container.querySelector('#tl-speed').addEventListener('change', (e) => setSpeed(parseFloat(e.target.value)));
    this.container.querySelector('#tl-scrub').addEventListener('input', (e) => setIdx(parseInt(e.target.value, 10)));
    this.container.querySelector('#tl-prev-haz').addEventListener('click', () => this._jumpHazard(-1));
    this.container.querySelector('#tl-next-haz').addEventListener('click', () => this._jumpHazard(1));

    on('session-loaded', () => this._onSessionLoaded());
    on('frame-changed', () => this.renderScrub());
    on('playback-changed', ({ playing }) => this._syncPlayIcon(playing));
    if (state.manifest) this._onSessionLoaded();
  }

  _syncPlayIcon(playing) {
    this.container.querySelector('#tl-play').innerHTML = icon(playing ? 'pause' : 'play');
    refreshIcons();
  }

  _jumpHazard(dir) {
    const events = allHazardEvents();
    if (!events.length) return;
    const future = dir > 0 ? events.filter((e) => e.frame > state.idx) : events.filter((e) => e.frame < state.idx).reverse();
    const target = future[0] || (dir > 0 ? events[0] : events[events.length - 1]);
    jumpToEvent(target);
  }

  _onSessionLoaded() {
    const scrub = this.container.querySelector('#tl-scrub');
    scrub.max = state.timeline.length - 1;
    this._renderMarkers();
    this.renderScrub();
  }

  _renderMarkers() {
    const wrap = this.container.querySelector('#tl-markers');
    const n = state.timeline.length;
    const pts = [];
    (state.manifest.node_events || []).forEach((ev) => {
      pts.push({ frame: ev.frame, kind: ev.kind === 'node_created' ? 'detection' : 'reassociated', payload: ev });
    });
    // "leaving the field of view" isn't logged server-side as a discrete
    // event; derive it by diffing each node's visible flag between
    // consecutive frames, which is exactly what the field means.
    const lastVisible = new Map();
    state.timeline.forEach((rec, idx) => {
      rec.nodes.forEach((node) => {
        const was = lastVisible.get(node.node_id);
        if (was === true && node.visible === false) pts.push({ frame: idx, kind: 'leave_frame', payload: { node_id: node.node_id } });
        lastVisible.set(node.node_id, node.visible);
      });
    });
    allHazardEvents().forEach((ev) => pts.push({ frame: ev.frame, kind: ev.severity, payload: ev }));

    wrap.innerHTML = pts
      .map((p) => `<div class="timeline-event-marker ${p.kind}" style="left:${(p.frame / Math.max(1, n - 1)) * 100}%" data-frame="${p.frame}" title="${p.kind} · frame ${p.frame}"></div>`)
      .join('');
    wrap.querySelectorAll('.timeline-event-marker').forEach((m) => m.addEventListener('click', () => setIdx(Number(m.dataset.frame))));
  }

  renderScrub() {
    const frame = currentFrame();
    if (!frame) return;
    this.container.querySelector('#tl-scrub').value = state.idx;
    const total = state.timeline[state.timeline.length - 1].t;
    this.container.querySelector('#tl-time').textContent = `${fmtS(frame.t)} / ${fmtS(total)}`;
    const pct = (state.idx / Math.max(1, state.timeline.length - 1)) * 100;
    this.container.querySelector('#tl-progress').style.width = `${pct}%`;
  }
}

// =====================================================================
// Perception Pipeline strip  (docs/UI.md section 5.E)
// =====================================================================
const PIPELINE_ORDER = ['camera', 'spark', 'sam31', 'lifting', 'memory_graph', 'ttc', 'tts'];

export class PipelineStatus {
  constructor(container) {
    this.container = container;
    on('static-data-loaded', () => this.render());
    this.render();
  }

  render() {
    const facts = state.systemFacts;
    if (!facts) { this.container.innerHTML = '<div class="inspector-empty">loading pipeline status…</div>'; return; }
    const byId = Object.fromEntries(facts.pipeline_components.map((c) => [c.id, c]));
    this.container.innerHTML = `<div class="pipeline-strip">${PIPELINE_ORDER.map((id, i) => {
      const c = byId[id];
      if (!c) return '';
      return `<div class="pipeline-stage" data-id="${id}">
        <div class="stage-top"><span class="stage-name">${c.name}</span><span class="stage-dot ${c.status}"></span></div>
        <div class="stage-desc">${(c.detail || '').slice(0, 86)}${(c.detail || '').length > 86 ? '…' : ''}</div>
        <div class="stage-latency">latency: not measured</div>
        ${i < PIPELINE_ORDER.length - 1 ? `<div class="pipeline-arrow">${icon('chevron-right')}</div>` : ''}
      </div>`;
    }).join('')}</div>`;
    this.container.querySelectorAll('.pipeline-stage').forEach((s) =>
      s.addEventListener('click', () => {
        const c = byId[s.dataset.id];
        showPopover(s, `
          <div class="pv-title">${icon('info')}${c.name}</div>
          <div class="pv-row"><span class="k">Status</span><span>${c.status}</span></div>
          ${c.model_id ? `<div class="pv-row"><span class="k">Model</span><span class="mono">${c.model_id}</span></div>` : ''}
          <div style="margin-top:6px;">${c.detail}</div>
          ${c.known_limitation ? `<div class="pv-warn">${c.known_limitation}</div>` : ''}
        `, c.id ?? s.dataset.id);
      })
    );
    refreshIcons();
  }
}

// =====================================================================
// Object Inspector  (shared by Overview / World Memory / Hazard Timeline)
// =====================================================================
export class ObjectInspector {
  constructor(container) {
    this.container = container;
    on('node-selected', () => this.render());
    on('frame-changed', () => this.render());
    on('session-loaded', () => this.render());
    this.render();
  }

  render() {
    const frame = currentFrame();
    if (!frame || state.selectedNodeId == null) {
      this.container.innerHTML = `<div class="inspector-empty">${icon('mouse-pointer-click')}<br>Select an object in the 3D view or camera feed to inspect its track.</div>`;
      refreshIcons();
      return;
    }
    const node = nodeById(state.selectedNodeId);
    if (!node) {
      this.container.innerHTML = `<div class="inspector-empty">Track #${state.selectedNodeId} has been evicted from memory (no observation for too long).</div>`;
      return;
    }
    const edge = frame.edges.find((e) => e.node_id === node.node_id);
    const speed = Math.hypot(...node.velocity_world);
    const dirDeg = speed > 0.05 ? (Math.atan2(node.velocity_world[0], node.velocity_world[2]) * 180) / Math.PI : null;
    const age = frame.t - node.last_seen;
    const relatedEvents = (state.manifest.node_events || []).filter((e) => e.node_id === node.node_id);

    this.container.innerHTML = `
      <div class="inspector-track"><span class="tid">#${node.node_id}</span><span class="badge ${node.visible ? 'safe' : 'neutral'}">${node.concept}</span>
        <span class="badge ${node.visible ? 'info' : 'neutral'}">${node.visible ? 'currently visible' : 'out of frame'}</span></div>
      <div class="inspector-field-grid">
        <div class="field"><span class="k">World position</span><span class="v">${node.p_world.map((v) => v.toFixed(2)).join(', ')} m</span></div>
        <div class="field"><span class="k">Velocity</span><span class="v">${speed.toFixed(2)} m/s</span></div>
        <div class="field"><span class="k">Direction</span><span class="v">${dirDeg != null ? dirDeg.toFixed(0) + '°' : 'stationary'}</span></div>
        <div class="field"><span class="k">Memory age</span><span class="v">${age <= 0 ? 'observed now' : age.toFixed(1) + 's since seen'}</span></div>
        <div class="field"><span class="k">Confidence</span><span class="v">${(node.confidence * 100).toFixed(0)}%</span></div>
        <div class="field"><span class="k">Observations</span><span class="v">${node.observation_count}</span></div>
        <div class="field"><span class="k">First seen</span><span class="v">t=${node.first_seen.toFixed(1)}s</span></div>
        <div class="field"><span class="k">Last seen</span><span class="v">t=${node.last_seen.toFixed(1)}s</span></div>
      </div>
      <div class="inspector-divider"></div>
      <div class="field"><span class="k">Association history</span></div>
      <div style="font-size:12px; color:var(--text-secondary); margin-top:6px;">
        ${relatedEvents.length ? relatedEvents.map((e) => `<div>t=${e.t.toFixed(1)}s &middot; ${e.kind}${e.out_of_view_s != null ? ` after ${e.out_of_view_s.toFixed(1)}s out of view` : ''}</div>`).join('') : 'no re-association events logged for this track'}
      </div>
      ${edge ? `<div class="inspector-divider"></div><div class="field"><span class="k">Related hazard event</span><span class="v">range ${edge.range_m.toFixed(1)}m, ${edge.ttc_s != null ? 'TTC ' + edge.ttc_s.toFixed(1) + 's' : 'not closing'}</span></div>` : ''}
    `;
    refreshIcons();
  }
}
