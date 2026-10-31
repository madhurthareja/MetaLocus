import { state, on, currentFrame, mostUrgentEdge, hazardSeverity } from '../state.js';
import { icon, refreshIcons } from '../icons.js';
import { WorldViewPanel } from '../worldpanel.js';
import { CameraFeed, HazardPanel, HudPreview, EventTimeline, PipelineStatus } from '../components.js';

export function renderShell(container) {
  container.innerHTML = `
    <div class="page-head">
      <div class="page-head-row">
        <div>
          <div class="page-title">Your surroundings, remembered.</div>
          <div class="page-subtitle">Persistent 3D awareness beyond the camera frame.</div>
        </div>
      </div>
      <div class="metrics-row" id="ov-metrics"></div>
    </div>

    <div class="overview-grid">
      <div class="col-stack">
        <div class="card world-view-card">
          <div class="card-header"><div><h3>Interactive 3D World View</h3><div class="sub">Rotate, zoom and click any object -- this is Locus's persistent world memory, not a replay of the camera</div></div></div>
          <div class="card-body" id="ov-world"></div>
        </div>
        <div class="card"><div class="card-body flush" id="ov-timeline"></div></div>
        <div class="card">
          <div class="card-header"><div><h3>Perception Pipeline</h3><div class="sub">Camera &rarr; Muse Spark &rarr; SAM 3.1 &rarr; Depth Lifting &rarr; World Memory &rarr; TTC &rarr; Audio</div></div></div>
          <div class="card-body flush" id="ov-pipeline"></div>
        </div>
      </div>
      <div class="col-stack">
        <div class="card"><div id="ov-camera"></div></div>
        <div class="card hazard-card state-safe">
          <div class="card-header"><h3>Active Hazard Panel</h3></div>
          <div class="card-body" id="ov-hazard"></div>
        </div>
        <div class="card">
          <div class="card-header"><div><h3>Ray-Ban Display HUD Preview</h3><div class="sub">What the wearer sees &middot; 600&times;600, true-black transparent</div></div></div>
          <div id="ov-hud"></div>
        </div>
      </div>
    </div>`;

  new WorldViewPanel(container.querySelector('#ov-world'), { dataset: state.manifest?.dataset, compact: false, showBanner: true });
  new EventTimeline(container.querySelector('#ov-timeline'));
  new PipelineStatus(container.querySelector('#ov-pipeline'));
  new CameraFeed(container.querySelector('#ov-camera'));
  new HazardPanel(container.querySelector('#ov-hazard'));
  new HudPreview(container.querySelector('#ov-hud'));

  const metrics = new MetricsRow(container.querySelector('#ov-metrics'));
}

class MetricsRow {
  constructor(container) {
    this.container = container;
    on('frame-changed', () => this.render());
    on('session-loaded', () => this.render());
    this.render();
  }

  render() {
    const frame = currentFrame();
    if (!frame) { this.container.innerHTML = ''; return; }
    const visible = frame.nodes.filter((n) => n.visible).length;
    const retained = frame.nodes.length - visible;
    const hazards = frame.edges.filter((e) => hazardSeverity(e) !== 'safe' && hazardSeverity(e) !== 'advisory');
    const worstEdge = mostUrgentEdge(frame.edges);
    const sev = hazardSeverity(worstEdge);
    const navLabel = sev === 'critical' ? 'CRITICAL HAZARD NEARBY' : sev === 'hazard' ? 'CAUTION ADVISED' : 'NO HAZARD DETECTED';
    const navBadge = sev === 'critical' ? 'critical' : sev === 'hazard' ? 'hazard' : 'safe';

    this.container.innerHTML = `
      <div class="card metric-card">
        <div class="metric-label">Tracked Objects</div>
        <div class="metric-value">${frame.nodes.length}</div>
        <div class="metric-detail">
          <div class="row"><span>Currently visible</span><span class="mono">${visible}</span></div>
          <div class="row"><span>Retained outside frame</span><span class="mono">${retained}</span></div>
        </div>
      </div>
      <div class="card metric-card">
        <div class="metric-label">Active Hazards</div>
        <div class="metric-value">${hazards.length}</div>
        <div class="metric-detail">
          <div class="row"><span>Severity</span><span class="badge ${navBadge}">${sev}</span></div>
        </div>
      </div>
      <div class="card metric-card">
        <div class="metric-label">Navigation Status</div>
        <div class="metric-value" style="font-size:18px;"><span class="badge ${navBadge}">${navLabel}</span></div>
        <div class="metric-detail"><div class="row" style="display:block; color:var(--text-secondary);">No hazard detected does not mean the route is safe.</div></div>
      </div>`;
  }
}
