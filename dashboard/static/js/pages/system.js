import { state, on } from '../state.js';
import { icon, refreshIcons } from '../icons.js';

export function renderShell(container) {
  container.innerHTML = `
    <div class="page-head">
      <div class="page-title">System</div>
      <div class="page-subtitle">Processing architecture, sensor calibration and coordinate-frame diagnostics.</div>
    </div>
    <div id="sys-body"><div class="empty-state">${icon('loader')}<div class="et">Loading system diagnostics&hellip;</div></div></div>`;
  refreshIcons();
  const renderAll = () => { if (state.systemFacts) fillBody(container.querySelector('#sys-body')); };
  on('static-data-loaded', renderAll);
  renderAll();
}

function fillBody(el) {
  const facts = state.systemFacts;
  const bf = facts.body_frame_validation;

  el.innerHTML = `
    <div class="card" style="margin-bottom:var(--space-4);">
      <div class="card-header"><h3>Processing Components</h3></div>
      <div class="card-body">
        <div class="components-grid">
          ${facts.pipeline_components.map((c) => `
            <div class="component-card">
              <div class="ctop"><span class="cname">${c.name}</span><span class="stage-dot ${c.status}"></span></div>
              ${c.model_id ? `<div class="mono" style="font-size:10.5px; color:var(--blue);">${c.model_id}</div>` : ''}
              <div class="cdetail">${c.detail}</div>
              ${c.known_limitation ? `<div class="cdetail" style="color:#8A5A10;">${icon('triangle-alert')} ${c.known_limitation}</div>` : ''}
              <div class="cfoot"><span class="badge ${c.status === 'real' ? 'safe' : 'demo'}">${c.status}</span><span>latency: not measured</span></div>
            </div>`).join('')}
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:var(--space-4);">
      <div class="card-header"><div><h3>Body-Frame Validation</h3><div class="sub">Demonstrates the body-frame design was validated against an independent camera, not merely assumed</div></div></div>
      <div class="card-body">
        <div class="frame-diagram" style="margin-bottom:var(--space-4);">
          <div class="frame-item"><div class="fname">F_ego</div><div class="fdesc">${bf.frames.camera_frame}</div></div>
          <div class="frame-item"><div class="fname">F_body</div><div class="fdesc">${bf.frames.body_frame}</div></div>
          <div class="frame-item"><div class="fname">F_world</div><div class="fdesc">${bf.frames.world_frame}</div></div>
        </div>
        <div class="notice-box" style="margin-bottom:var(--space-3);">${icon('check-circle')}<span><strong>Invariant tested:</strong> ${bf.frames.invariant_tested}</span></div>
        <div style="display:flex; gap:var(--space-4); align-items:center; margin-bottom:var(--space-3);">
          <div style="text-align:center;"><div class="metric-value" style="font-size:28px;">${bf.transform_consistency_check.result_median_heading_disagreement_degrees}&deg;</div><div class="metric-label">median heading disagreement</div></div>
          <div style="flex:1; font-size:12.5px; color:var(--ink); line-height:1.6;">${bf.transform_consistency_check.what}<br><span style="color:var(--text-secondary);">${bf.transform_consistency_check.result_note}</span></div>
        </div>
        <h4 style="font-size:11.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--text-secondary); margin-bottom:8px;">Bugs found by this check</h4>
        <ul style="margin:0 0 var(--space-3); padding-left:18px; font-size:12px; color:var(--ink); line-height:1.7;">
          ${bf.transform_consistency_check.bugs_found_by_this_check.map((b) => `<li>${b}</li>`).join('')}
        </ul>
        <div class="notice-box warn">${icon('triangle-alert')}<span><strong>Unresolved caveat (stated plainly, not hidden):</strong> ${bf.transform_consistency_check.unresolved_caveat}</span></div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Real SANPO Session Summary</h3></div>
      <div class="card-body">
        <div class="kv-list">
          <div class="kv-row"><span class="k">Session</span><span class="v">${facts.real_sanpo_session_summary.session_id}</span></div>
          <div class="kv-row"><span class="k">Frames</span><span class="v">${facts.real_sanpo_session_summary.n_frames} @ ${facts.real_sanpo_session_summary.fps}fps</span></div>
          <div class="kv-row"><span class="k">Concepts</span><span class="v">${facts.real_sanpo_session_summary.concepts.join(', ')}</span></div>
          <div class="kv-row"><span class="k">Raw detections &rarr; lifted to 3D</span><span class="v">${facts.real_sanpo_session_summary.total_detections} &rarr; ${facts.real_sanpo_session_summary.total_lifted}</span></div>
          <div class="kv-row"><span class="k">Final node count</span><span class="v">${facts.real_sanpo_session_summary.final_graph_summary.n_nodes}</span></div>
        </div>
        <div class="notice-box" style="margin-top:var(--space-3);">${icon('info')}<span>${facts.latency_disclaimer}</span></div>
      </div>
    </div>`;
  refreshIcons();
}
