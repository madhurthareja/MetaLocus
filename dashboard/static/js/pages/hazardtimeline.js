import { state, on, currentFrame, allHazardEvents, jumpToEvent, nodeById } from '../state.js';
import { icon, refreshIcons } from '../icons.js';
import { WorldViewPanel } from '../worldpanel.js';
import { CameraFeed, EventTimeline } from '../components.js';

const RECON_STEPS = [
  ['scan', 'Initial detection', 'Muse Spark concept proposed, SAM 3.1 session running on that concept'],
  ['scissors', 'Segmentation', 'SAM 3.1 Object Multiplex mask produced for this instance'],
  ['box', 'Depth lifting', 'Median masked depth lifted the mask to a 3D camera-frame point'],
  ['globe', 'World-frame association', 'Transformed to world coordinates, matched or created a persistent node'],
  ['eye-off', 'Left camera frame', 'Detection stopped; node retained in memory rather than deleted'],
  ['brain', 'Memory retention', 'Confidence decays by half-life; position held at last observation'],
  ['route', 'Collision prediction', 'Geometric time-to-collision computed from range and closing speed'],
  ['megaphone', 'Warning generation', 'Directive composed and spoken, spatial tone panned toward bearing'],
];

export function renderShell(container) {
  container.innerHTML = `
    <div class="page-head">
      <div class="page-title">Hazard Timeline</div>
      <div class="page-subtitle">Research-grade reconstruction of every hazard-grade event in the loaded session.</div>
    </div>
    <div class="card" style="margin-bottom:var(--space-4);"><div class="card-body flush" id="ht-timeline"></div></div>
    <div class="overview-grid" style="grid-template-columns: 1fr 1.3fr;">
      <div class="card">
        <div class="card-header"><div><h3>Hazard Events</h3><div class="sub" id="ht-count"></div></div></div>
        <div class="card-body flush"><div style="max-height:600px; overflow-y:auto;"><table class="data-table" id="ht-table">
          <thead><tr><th>ID</th><th>t</th><th>Object</th><th>Severity</th><th>TTC</th><th>Out-of-view</th></tr></thead>
          <tbody id="ht-rows"></tbody>
        </table></div></div>
      </div>
      <div class="col-stack">
        <div class="card"><div class="card-header"><h3>Event Reconstruction</h3></div><div class="card-body" id="ht-recon"></div></div>
        <div class="card"><div id="ht-camera"></div></div>
        <div class="card"><div class="card-body" id="ht-world" style="padding:0;"></div></div>
      </div>
    </div>`;

  new EventTimeline(container.querySelector('#ht-timeline'));
  new CameraFeed(container.querySelector('#ht-camera'));
  new WorldViewPanel(container.querySelector('#ht-world'), { dataset: state.manifest?.dataset, compact: true, showBanner: false });
  new EventTable(container.querySelector('#ht-rows'), container.querySelector('#ht-count'));
  new Reconstruction(container.querySelector('#ht-recon'));
}

class EventTable {
  constructor(tbody, countEl) {
    this.tbody = tbody;
    this.countEl = countEl;
    this.selected = null;
    on('session-loaded', () => this.render());
    this.render();
  }
  render() {
    if (!state.manifest) return;
    const events = allHazardEvents();
    this.events = events;
    this.countEl.textContent = `${events.length} events · TTC threshold 2.0s (critical) / 5.0s (advisory)`;
    this.tbody.innerHTML = events.map((e, i) => {
      const node = state.timeline[e.frame]?.nodes.find((n) => n.node_id === e.node_id);
      const outOfView = node && !node.visible ? (state.timeline[e.frame].t - node.last_seen).toFixed(1) + 's' : 'in frame';
      return `<tr data-i="${i}" tabindex="0" class="${this.selected === i ? 'selected' : ''}">
        <td class="mono">EVT-${String(i + 1).padStart(3, '0')}</td>
        <td class="mono">${e.t.toFixed(1)}s</td>
        <td>#${e.node_id} ${e.concept}</td>
        <td><span class="badge ${e.severity === 'critical' ? 'critical' : 'hazard'}">${e.severity}</span></td>
        <td class="mono">${e.edge.ttc_s != null ? e.edge.ttc_s.toFixed(1) + 's' : '--'}</td>
        <td class="mono">${outOfView}</td>
      </tr>`;
    }).join('') || `<tr><td colspan="6"><div class="inspector-empty">no hazard-grade events in this session</div></td></tr>`;
    this.tbody.querySelectorAll('tr[data-i]').forEach((row) => row.addEventListener('click', () => {
      const i = Number(row.dataset.i);
      this.selected = i;
      this.tbody.querySelectorAll('tr[data-i]').forEach((r) => r.classList.toggle('selected', Number(r.dataset.i) === i));
      jumpToEvent(this.events[i]);
    }));
  }
}

class Reconstruction {
  constructor(container) {
    this.container = container;
    on('frame-changed', () => this.render());
    on('session-loaded', () => this.render());
    this.render();
  }
  render() {
    const frame = currentFrame();
    if (!frame) { this.container.innerHTML = '<div class="inspector-empty">no session loaded</div>'; return; }
    const node = state.selectedNodeId != null ? nodeById(state.selectedNodeId) : null;
    const edge = node ? frame.edges.find((e) => e.node_id === node.node_id) : null;
    const warningGenerated = !!edge;
    const leftFrame = node && !node.visible;

    this.container.innerHTML = `
      <div style="font-size:11.5px; color:var(--text-secondary); margin-bottom:10px;">
        ${node ? `Reconstructing track #${node.node_id} (${node.concept}) at t=${frame.t.toFixed(1)}s` : 'Select a hazard event or an object to see its step-by-step reconstruction.'}
      </div>
      ${RECON_STEPS.map(([ic, name, desc], i) => {
        let occurred = !!node;
        if (i === 4) occurred = leftFrame; // "left camera frame"
        if (i === 5) occurred = leftFrame;
        if (i >= 6) occurred = warningGenerated;
        return `<div style="display:flex; gap:10px; padding:7px 0; ${i < RECON_STEPS.length - 1 ? 'border-bottom:1px solid var(--border);' : ''}">
          <div style="width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; background:${occurred ? 'var(--info-bg)' : 'var(--surface)'}; color:${occurred ? 'var(--blue)' : 'var(--text-secondary)'};">${icon(ic)}</div>
          <div><div style="font-size:12px; font-weight:600; color:${occurred ? 'var(--ink)' : 'var(--text-secondary)'};">${name}</div><div style="font-size:11px; color:var(--text-secondary);">${desc}</div></div>
        </div>`;
      }).join('')}`;
    refreshIcons();
  }
}
