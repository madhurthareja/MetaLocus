import { state, on, currentFrame } from '../state.js';
import { icon, refreshIcons } from '../icons.js';
import { WorldViewPanel } from '../worldpanel.js';
import { ObjectInspector } from '../components.js';

export function renderShell(container) {
  container.innerHTML = `
    <div class="page-head">
      <div class="page-title">World Memory</div>
      <div class="page-subtitle">The persistent 3D memory graph -- a world model, not an accumulation of frame-level detections.</div>
    </div>
    <div class="overview-grid" style="grid-template-columns: 1.7fr 1fr;">
      <div class="col-stack">
        <div class="card">
          <div class="card-header">
            <div><h3>Memory Graph</h3><div class="sub">Filter by concept or memory state, inspect any track</div></div>
            <div style="display:flex; gap:8px;">
              <select class="hdr-select" id="wm-filter-state"><option value="all">All states</option><option value="visible">Visible only</option><option value="memory">Memory only</option></select>
              <select class="hdr-select" id="wm-filter-concept"><option value="all">All concepts</option></select>
            </div>
          </div>
          <div class="card-body" id="wm-world"></div>
        </div>
      </div>
      <div class="col-stack">
        <div class="card">
          <div class="card-header"><h3>Object Inspector</h3></div>
          <div class="card-body" id="wm-inspector"></div>
        </div>
        <div class="card">
          <div class="card-header"><div><h3>Memory Operations</h3><div class="sub">The actual mechanisms, not a frame buffer</div></div></div>
          <div class="card-body" id="wm-ops"></div>
        </div>
        <div class="card">
          <div class="card-header"><h3>Operations Log</h3></div>
          <div class="card-body flush" id="wm-log"></div>
        </div>
      </div>
    </div>`;

  const panel = new WorldViewPanel(container.querySelector('#wm-world'), { dataset: state.manifest?.dataset, compact: false, showBanner: false });
  new ObjectInspector(container.querySelector('#wm-inspector'));
  renderOpsExplainer(container.querySelector('#wm-ops'));
  new OpsLog(container.querySelector('#wm-log'));
  new ConceptFilter(container, panel);
}

function renderOpsExplainer(el) {
  const ops = [
    { icon: 'timer-reset', name: 'Temporal decay', desc: 'Confidence halves every 8s a node goes unobserved -- src/locus/memory/decay.py.' },
    { icon: 'trash-2', name: 'Node eviction', desc: 'A node with no observation for 20s is removed from the graph entirely.' },
    { icon: 'move-3d', name: 'Spatial re-association', desc: 'A new detection within 1.0m of an existing same-concept node is merged into it, not created as a new track.' },
    { icon: 'git-merge', name: 'Duplicate-track merging', desc: "Re-checked after every observation (not just at creation) -- fixed a real bug where SAM 3.1's tracker emitted two instance IDs for one physical object." },
  ];
  el.innerHTML = ops.map((o) => `<div style="display:flex; gap:10px; margin-bottom:14px;">
    <div style="color:var(--blue); flex-shrink:0;">${icon(o.icon)}</div>
    <div><div style="font-weight:600; font-size:12.5px; color:var(--ink);">${o.name}</div><div style="font-size:11.5px; color:var(--text-secondary); line-height:1.5;">${o.desc}</div></div>
  </div>`).join('');
  refreshIcons();
}

class OpsLog {
  constructor(container) {
    this.container = container;
    on('session-loaded', () => this.render());
    on('frame-changed', () => this.render());
    if (state.manifest) this.render();
  }
  render() {
    if (!state.manifest) return;
    const frame = currentFrame();
    const events = (state.manifest.node_events || []).filter((e) => e.frame <= (frame?.frame ?? 0)).slice(-40).reverse();
    this.container.innerHTML = `<div style="max-height:260px; overflow-y:auto;">` + (events.length
      ? events.map((e) => `<div style="padding:8px 16px; border-bottom:1px solid var(--border); font-size:11.5px;">
          <span class="mono" style="color:var(--text-secondary);">t=${e.t.toFixed(1)}s</span>
          <span style="margin-left:6px;">${e.kind === 'node_created' ? `new track <b>#${e.node_id}</b> (${e.concept})` : `track <b>#${e.node_id}</b> (${e.concept}) re-associated after ${e.out_of_view_s?.toFixed(1)}s out of view`}</span>
        </div>`).join('')
      : `<div class="inspector-empty">no memory operations logged yet at this point in the timeline</div>`) + `</div>`;
  }
}

class ConceptFilter {
  constructor(container, panel) {
    this.panel = panel;
    this.stateSel = container.querySelector('#wm-filter-state');
    this.conceptSel = container.querySelector('#wm-filter-concept');
    const populate = (manifest) => {
      this.conceptSel.innerHTML = '<option value="all">All concepts</option>' + manifest.concepts.map((c) => `<option value="${c}">${c}</option>`).join('');
    };
    on('session-loaded', ({ manifest }) => populate(manifest));
    if (state.manifest) populate(state.manifest);
    this.stateSel.addEventListener('change', () => this.panel.view.setFilter({ state: this.stateSel.value }));
    this.conceptSel.addEventListener('change', () => this.panel.view.setFilter({ concept: this.conceptSel.value }));
  }
}
