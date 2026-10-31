import { state, on } from '../state.js';
import { icon, refreshIcons } from '../icons.js';

const FAILURE_CATEGORIES = [
  'False positive warnings', 'False negative hazards', 'Track fragmentation',
  'Incorrect spatial associations', 'Stale memory', 'Depth estimation errors', 'Camera calibration issues',
];

export function renderShell(container) {
  container.innerHTML = `
    <div class="page-head">
      <div class="page-title">Experiment Lab</div>
      <div class="page-subtitle">Scientific evaluation of the persistent-memory thesis against stateless and intermediate baselines.</div>
    </div>
    <div id="el-body"><div class="empty-state">${icon('loader')}<div class="et">Loading evaluation data&hellip;</div></div></div>`;
  refreshIcons();

  const renderAll = () => { if (state.ablation && state.reproducibility) fillBody(container.querySelector('#el-body')); };
  on('static-data-loaded', renderAll);
  renderAll();
}

function fillBody(el) {
  const ab = state.ablation;
  const repro = state.reproducibility;
  const A = ab.conditions.A_stateless, D = ab.conditions.D_mask;

  el.innerHTML = `
    <div class="hero-stat-banner" style="margin-bottom:var(--space-4);">
      <div class="hero-stat"><div class="hv">${(A.out_of_view_answer_rate * 100).toFixed(0)}%</div><div class="hl">A: Stateless</div></div>
      <div class="hero-arrow">${icon('arrow-right')}</div>
      <div class="hero-stat"><div class="hv">${(D.out_of_view_answer_rate * 100).toFixed(0)}%</div><div class="hl">D: Persistent memory</div></div>
      <div class="hero-caption">Out-of-view answer rate &mdash; can the system correctly answer "where is it?" once an object has left the camera frame.
      <br><strong>Protocol:</strong> ${ab.evaluation_protocol.description} Denominator: ${A.out_of_view_n_queries} pooled out-of-view questions across ${ab.evaluation_protocol.n_scenarios} synthetic scenarios (${ab.evaluation_protocol.scenarios.join(', ')}).</div>
    </div>

    <div class="notice-box" style="margin-bottom:var(--space-4);">${icon('info')}<span>${ab.evaluation_protocol.why_synthetic}</span></div>

    <div class="card" style="margin-bottom:var(--space-4);">
      <div class="card-header"><h3>A. Ablation Studio</h3></div>
      <div class="card-body">
        <div class="condition-grid">${Object.entries(ab.conditions).map(([key, c]) => `
          <div class="condition-card ${key === 'D_mask' ? 'full' : ''}">
            <div class="cname">${c.label}</div>
            <div class="cdesc">${c.description}</div>
            <div class="cmetrics">
              <div class="cmetric"><span>In-view answer rate</span><span class="mv">${(c.in_view_answer_rate * 100).toFixed(0)}%</span></div>
              <div class="cmetric"><span>Out-of-view answer rate</span><span class="mv">${(c.out_of_view_answer_rate * 100).toFixed(0)}%</span></div>
              <div class="cmetric"><span>Bearing error (clock-hrs)</span><span class="mv">${c.out_of_view_mean_bearing_error_hours != null ? c.out_of_view_mean_bearing_error_hours.toFixed(2) : '--'}</span></div>
            </div>
          </div>`).join('')}
        </div>
        <div style="height:1px; background:var(--border); margin:var(--space-4) 0;"></div>
        <div class="bar-chart">
          ${Object.entries(ab.conditions).map(([key, c]) => `
            <div class="bar-row">
              <div class="label">${c.label.split(':')[0]}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${c.out_of_view_answer_rate * 100}%; background:${key === 'D_mask' ? 'var(--safe)' : key === 'A_stateless' ? 'var(--critical)' : 'var(--blue)'}"></div></div>
              <div class="pct">${(c.out_of_view_answer_rate * 100).toFixed(0)}%</div>
            </div>`).join('')}
        </div>
        <div class="notice-box warn" style="margin-top:var(--space-4);">${icon('triangle-alert')}<span>Not implemented in this codebase, shown honestly rather than fabricated: ${Object.keys(ab.metrics_not_implemented).join(', ')}. ${ab.metrics_not_implemented.hazard_detection_rate || ''}</span></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:var(--space-4);">
      <div class="card-header"><h3>B. Dataset Evaluation</h3><div class="sub">Same pipeline, two different sensor and depth modalities</div></div>
      <div class="card-body">
        <div class="dataset-compare-grid">
          <div class="card dataset-card" style="box-shadow:none;">
            <div class="card-header" style="border-bottom:none;"><h3>SANPO <span class="badge safe" style="margin-left:6px;">primary</span></h3></div>
            <div class="card-body" style="padding-top:0;">
              <div class="dstat"><span class="k">Role</span><span class="v">${repro.dataset_splits.sanpo.role}</span></div>
              <div class="dstat"><span class="k">Source</span><span class="v" style="font-weight:400; font-family:inherit;">${repro.dataset_splits.sanpo.source}</span></div>
              <div class="dstat"><span class="k">Dense depth coverage</span><span class="v">${repro.camera_calibration.sanpo_dense_depth_coverage_pct}%</span></div>
              <div class="dstat"><span class="k">Head camera</span><span class="v">${repro.sensor_configuration.sanpo.camera_head.model}, ${repro.sensor_configuration.sanpo.camera_head.stereo_baseline_mm.toFixed(1)}mm baseline</span></div>
              <div class="dstat"><span class="k">Chest camera</span><span class="v">${repro.sensor_configuration.sanpo.camera_chest.model}, ${repro.sensor_configuration.sanpo.camera_chest.stereo_baseline_mm.toFixed(1)}mm baseline</span></div>
              <div class="dstat"><span class="k">Real run</span><span class="v" id="el-sanpo-run"></span></div>
            </div>
          </div>
          <div class="card dataset-card" style="box-shadow:none;">
            <div class="card-header" style="border-bottom:none;"><h3>AEA <span class="badge neutral" style="margin-left:6px;">secondary</span></h3></div>
            <div class="card-body" style="padding-top:0;">
              <div class="dstat"><span class="k">Role</span><span class="v">${repro.dataset_splits.aea.role}</span></div>
              <div class="dstat"><span class="k">Sequences evaluated</span><span class="v">${repro.dataset_splits.aea.n_sequences}</span></div>
              <div class="dstat"><span class="k">Sparse depth coverage</span><span class="v">${repro.camera_calibration.aea_sparse_depth_coverage_pct}%</span></div>
              <div class="dstat"><span class="k">Camera</span><span class="v">${repro.sensor_configuration.aea.camera}</span></div>
              <div class="dstat"><span class="k">Distortion model</span><span class="v">${repro.sensor_configuration.aea.distortion_model}</span></div>
              <div class="dstat"><span class="k">Depth source</span><span class="v" style="font-weight:400; font-family:inherit;">${repro.camera_calibration.aea_depth_source}</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:var(--space-4);">
      <div class="card-header"><h3>C. Failure Analysis</h3></div>
      <div class="filter-row">
        <select disabled><option>Dataset: all</option></select>
        <select disabled><option>Object type: all</option></select>
        <select disabled><option>Category: all</option></select>
      </div>
      <div class="card-body">
        <div class="notice-box warn">${icon('triangle-alert')}<span>No real failure instances are logged by the current pipeline (no false-positive/false-negative tracking is implemented in src/locus/eval/). Showing the planned category list rather than inventing example failures.</span></div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:var(--space-3);">
          ${FAILURE_CATEGORIES.map((c) => `<span class="badge neutral">${c}</span>`).join('')}
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>D. Reproducibility</h3></div>
      <div class="card-body">
        <div class="repro-grid">
          <div>
            <h4 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px;">Model versions</h4>
            <div class="kv-list">
              <div class="kv-row"><span class="k">Concept proposal</span><span class="v">${repro.model_versions.concept_proposal.model_id}</span></div>
              <div class="kv-row"><span class="k">Segmentation</span><span class="v">${repro.model_versions.segmentation.version} (fa3 off)</span></div>
              <div class="kv-row"><span class="k">Speech</span><span class="v">${repro.model_versions.speech.model_id}</span></div>
            </div>
            <h4 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.04em; margin:16px 0 8px;">Thresholds</h4>
            <div class="kv-list">
              ${Object.entries(repro.thresholds).map(([k, v]) => `<div class="kv-row"><span class="k">${k.replace(/_/g, ' ')}</span><span class="v">${v}</span></div>`).join('')}
              <div class="kv-row"><span class="k">edge radius (SANPO demo run)</span><span class="v">20.0 m (CLI override for outdoor range)</span></div>
            </div>
          </div>
          <div>
            <h4 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px;">Dataset splits</h4>
            <div class="kv-list">
              <div class="kv-row"><span class="k">SANPO sessions</span><span class="v">${repro.dataset_splits.sanpo.sessions_used.length}</span></div>
              <div class="kv-row"><span class="k">AEA sequences</span><span class="v">${repro.dataset_splits.aea.n_sequences}</span></div>
              <div class="kv-row"><span class="k">Synthetic scenarios</span><span class="v">${repro.dataset_splits.synthetic.scenarios.length}</span></div>
            </div>
            <h4 style="font-size:12px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.04em; margin:16px 0 8px;">Camera calibration</h4>
            <div class="kv-list">
              <div class="kv-row"><span class="k">AEA native sensor</span><span class="v">${repro.sensor_configuration.aea.native_sensor_resolution_px}px</span></div>
              <div class="kv-row"><span class="k">AEA undistortion</span><span class="v" style="font-weight:400; font-family:inherit; text-align:left; max-width:220px;">${repro.sensor_configuration.aea.undistortion}</span></div>
              <div class="kv-row"><span class="k">SANPO distortion</span><span class="v">${repro.sensor_configuration.sanpo.distortion}</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  const run = state.systemFacts?.real_sanpo_session_summary;
  const runEl = el.querySelector('#el-sanpo-run');
  if (run && runEl) runEl.textContent = `${run.n_frames}f, ${run.total_detections}→${run.total_lifted} lifted, ${run.final_graph_summary.n_nodes} nodes`;
  refreshIcons();
}
