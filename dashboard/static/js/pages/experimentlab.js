import { state, on } from '../state.js';
import { icon, refreshIcons } from '../icons.js';
import { showPopover } from '../components.js';

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
    <div id="el-body"><div class="empty-state">${icon('loader', 'class="spin"')}<div class="et">Loading evaluation data&hellip;</div></div></div>`;
  refreshIcons();

  const renderAll = () => { if (state.ablation && state.reproducibility) fillBody(container.querySelector('#el-body')); };
  on('static-data-loaded', renderAll);
  renderAll();
}

function fillBody(el) {
  const ab = state.ablation;
  const repro = state.reproducibility;
  const A = ab.conditions.A_stateless, D = ab.conditions.D_mask;
  const run = state.systemFacts?.real_sanpo_session_summary;

  el.innerHTML = `
    <div class="hero-stat-banner" style="margin-bottom:var(--space-4);">
      <div class="hero-stat"><div class="hv">${(A.out_of_view_answer_rate * 100).toFixed(0)}%</div><div class="hl">A: Stateless</div></div>
      <div class="hero-arrow">${icon('arrow-right')}</div>
      <div class="hero-stat"><div class="hv">${(D.out_of_view_answer_rate * 100).toFixed(0)}%</div><div class="hl">D: Persistent memory</div></div>
      <div class="hero-caption">Out-of-view answer rate &mdash; can the system say "where is it?" once an object leaves the camera frame.<br>
        <button class="detail-trigger" id="el-protocol-info" style="color:#9FB4CF;">${icon('circle-help')} How this was measured &amp; why synthetic data</button>
      </div>
    </div>

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
              <div style="font-size:11.5px; color:var(--text-secondary); margin-bottom:12px;">${repro.dataset_splits.sanpo.role} &middot; ${repro.sensor_configuration.sanpo.camera_head.model} (head) + ${repro.sensor_configuration.sanpo.camera_chest.model} (chest)</div>
              <div class="meter-row">
                <div class="meter-label"><span>Dense depth coverage</span><span class="mv">${repro.camera_calibration.sanpo_dense_depth_coverage_pct}%</span></div>
                <div class="meter-track"><div class="meter-fill" style="width:${repro.camera_calibration.sanpo_dense_depth_coverage_pct}%"></div></div>
              </div>
              <div class="stat-tile-grid" style="margin-top:12px;">
                <div class="stat-tile"><div class="stv">${repro.sensor_configuration.sanpo.camera_head.stereo_baseline_mm.toFixed(1)}mm</div><div class="stl">Head baseline</div></div>
                <div class="stat-tile"><div class="stv">${repro.sensor_configuration.sanpo.camera_chest.stereo_baseline_mm.toFixed(1)}mm</div><div class="stl">Chest baseline</div></div>
                <div class="stat-tile"><div class="stv">${run ? run.n_frames : '--'}</div><div class="stl">Real run frames</div></div>
              </div>
              <button class="detail-trigger" id="el-sanpo-detail" style="margin-top:10px;">${icon('circle-help')} Source &amp; real-run detail</button>
            </div>
          </div>
          <div class="card dataset-card" style="box-shadow:none;">
            <div class="card-header" style="border-bottom:none;"><h3>AEA <span class="badge neutral" style="margin-left:6px;">secondary</span></h3></div>
            <div class="card-body" style="padding-top:0;">
              <div style="font-size:11.5px; color:var(--text-secondary); margin-bottom:12px;">${repro.dataset_splits.aea.role} &middot; ${repro.sensor_configuration.aea.camera}</div>
              <div class="meter-row">
                <div class="meter-label"><span>Sparse depth coverage</span><span class="mv">${repro.camera_calibration.aea_sparse_depth_coverage_pct}%</span></div>
                <div class="meter-track"><div class="meter-fill" style="width:${repro.camera_calibration.aea_sparse_depth_coverage_pct}%"></div></div>
              </div>
              <div class="stat-tile-grid" style="margin-top:12px;">
                <div class="stat-tile"><div class="stv">${repro.dataset_splits.aea.n_sequences}</div><div class="stl">Sequences evaluated</div></div>
                <div class="stat-tile"><div class="stv">${repro.sensor_configuration.aea.native_sensor_resolution_px}</div><div class="stl">Native res. (px)</div></div>
              </div>
              <button class="detail-trigger" id="el-aea-detail" style="margin-top:10px;">${icon('circle-help')} Distortion &amp; depth-source detail</button>
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
      <div class="card-header">
        <div><h3>D. Reproducibility</h3><div class="sub">${repro.model_versions.segmentation.version} &middot; ${repro.dataset_splits.sanpo.sessions_used.length} SANPO session(s) &middot; ${repro.dataset_splits.aea.n_sequences} AEA sequence(s) &middot; ${repro.dataset_splits.synthetic.scenarios.length} synthetic scenario(s)</div></div>
        <button class="disclosure-toggle" id="el-repro-toggle">${icon('chevron-down')} Show full configuration</button>
      </div>
      <div class="card-body disclosure-body" id="el-repro-body" style="padding:0;">
        <div class="repro-grid" style="padding:var(--space-4);">
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

  el.querySelector('#el-protocol-info').addEventListener('click', (e) => {
    showPopover(e.currentTarget, `
      <div class="pv-title">${icon('flask-conical')}Evaluation protocol</div>
      <div style="margin-top:4px;">${ab.evaluation_protocol.description} Denominator: ${A.out_of_view_n_queries} pooled out-of-view questions across ${ab.evaluation_protocol.n_scenarios} synthetic scenarios (${ab.evaluation_protocol.scenarios.join(', ')}).</div>
      <div class="pv-warn">${ab.evaluation_protocol.why_synthetic}</div>
    `, 'protocol');
  });

  el.querySelector('#el-sanpo-detail').addEventListener('click', (e) => {
    showPopover(e.currentTarget, `
      <div class="pv-title">${icon('database')}SANPO detail</div>
      <div class="pv-row"><span class="k">Source</span><span>${repro.dataset_splits.sanpo.source}</span></div>
      ${run ? `
      <div class="pv-row"><span class="k">Real run</span><span>${run.n_frames} frames</span></div>
      <div class="pv-row"><span class="k">Detections &rarr; lifted</span><span>${run.total_detections} &rarr; ${run.total_lifted}</span></div>
      <div class="pv-row"><span class="k">Final nodes</span><span>${run.final_graph_summary.n_nodes}</span></div>` : ''}
    `, 'sanpo-detail');
  });

  el.querySelector('#el-aea-detail').addEventListener('click', (e) => {
    showPopover(e.currentTarget, `
      <div class="pv-title">${icon('database')}AEA detail</div>
      <div class="pv-row"><span class="k">Distortion model</span><span>${repro.sensor_configuration.aea.distortion_model}</span></div>
      <div class="pv-row"><span class="k">Depth source</span><span>${repro.camera_calibration.aea_depth_source}</span></div>
      <div class="pv-row"><span class="k">Undistortion</span><span>${repro.sensor_configuration.aea.undistortion}</span></div>
    `, 'aea-detail');
  });

  const reproToggle = el.querySelector('#el-repro-toggle');
  const reproBody = el.querySelector('#el-repro-body');
  reproToggle.addEventListener('click', () => {
    const open = reproBody.classList.toggle('open');
    reproToggle.classList.toggle('open', open);
    reproToggle.innerHTML = `${icon('chevron-down')} ${open ? 'Hide full configuration' : 'Show full configuration'}`;
    refreshIcons();
  });

  refreshIcons();
}
