// Shared data/event model. Every page/component reads from this single
// store and reacts to its events -- per docs/UI.md section 12, "all major
// dashboard components should consume a shared event/timeline data source
// so that camera frames, world memory, hazards and HUD output remain
// synchronized." Nothing in here renders DOM; it only owns state + timing.

export const bus = new EventTarget();
export function on(name, fn) { bus.addEventListener(name, (e) => fn(e.detail)); }
function emit(name, detail) { bus.dispatchEvent(new CustomEvent(name, { detail })); }

export const CRITICAL_TTC_S = 2.0;   // matches reproducibility.json hazard.ttc_threshold_seconds
export const HAZARD_TTC_S = 5.0;     // earlier advisory lead time, same threshold the HUD/audio already use

export const state = {
  sessions: [],
  session: null,
  manifest: null,
  timeline: [],
  idx: 0,
  playing: false,
  speed: 1,
  showMasks: true,
  selectedNodeId: null,
  hoveredNodeId: null,
  startedAt: null,

  ablation: null,
  reproducibility: null,
  systemFacts: null,

  audioMuted: false,
};

export function currentFrame() {
  return state.timeline[state.idx] || null;
}

export function hazardSeverity(edge) {
  if (!edge || edge.ttc_s == null) return edge ? 'advisory' : 'safe';
  if (edge.ttc_s < CRITICAL_TTC_S) return 'critical';
  if (edge.ttc_s < HAZARD_TTC_S) return 'hazard';
  return 'advisory';
}

export function mostUrgentEdge(edges) {
  if (!edges || !edges.length) return null;
  const withTtc = edges.filter((e) => e.ttc_s != null);
  if (withTtc.length) return withTtc.reduce((a, b) => (a.ttc_s < b.ttc_s ? a : b));
  return edges.reduce((a, b) => (a.range_m < b.range_m ? a : b));
}

export function clockHour(bearingDeg) {
  const h = Math.round((((bearingDeg % 360) + 360) % 360) / 30) % 12;
  return h === 0 ? 12 : h;
}

export function nodeById(id) {
  const frame = currentFrame();
  if (!frame) return null;
  return frame.nodes.find((n) => n.node_id === id) || null;
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export async function loadStaticData() {
  const [ablation, reproducibility, systemFacts] = await Promise.all([
    fetchJSON('/data/ablation_results.json').catch(() => null),
    fetchJSON('/data/reproducibility.json').catch(() => null),
    fetchJSON('/data/system_facts.json').catch(() => null),
  ]);
  state.ablation = ablation;
  state.reproducibility = reproducibility;
  state.systemFacts = systemFacts;
  emit('static-data-loaded', {});
}

export async function loadSessionList() {
  state.sessions = await fetchJSON('/api/sessions');
  emit('sessions-loaded', state.sessions);
  return state.sessions;
}

export async function loadSession(name) {
  emit('session-loading', { name });
  const manifest = await fetchJSON(`/api/timeline/${name}`);
  state.session = name;
  state.manifest = manifest;
  state.timeline = manifest.timeline;
  state.idx = 0;
  state.selectedNodeId = null;
  state.startedAt = null;
  emit('session-loaded', { name, manifest });
  emit('frame-changed', { idx: 0, frame: currentFrame() });
}

export function setIdx(i) {
  state.idx = Math.max(0, Math.min(state.timeline.length - 1, i));
  emit('frame-changed', { idx: state.idx, frame: currentFrame() });
}

export function play() {
  if (!state.timeline.length) return;
  if (state.idx >= state.timeline.length - 1) state.idx = 0;
  state.playing = true;
  state._lastTick = null;
  emit('playback-changed', { playing: true });
  requestAnimationFrame(playLoop);
}

export function pause() {
  state.playing = false;
  if (_directiveAudio) _directiveAudio.pause(); // a clip already playing must stop too, not just frame advancement
  emit('playback-changed', { playing: false });
}

export function togglePlay() {
  state.playing ? pause() : play();
}

export function setSpeed(s) {
  state.speed = s;
  emit('speed-changed', { speed: s });
}

export function setShowMasks(v) {
  state.showMasks = v;
  emit('masks-toggled', { showMasks: v });
}

export function selectNode(id) {
  state.selectedNodeId = id;
  emit('node-selected', { nodeId: id });
}

export function hoverNode(id) {
  state.hoveredNodeId = id;
  emit('node-hovered', { nodeId: id });
}

function playLoop(ts) {
  if (!state.playing) return;
  if (state._lastTick == null) state._lastTick = ts;
  const fps = (state.manifest?.fps || 15) * state.speed;
  const dtMs = ts - state._lastTick;
  if (dtMs > 1000 / fps) {
    state._lastTick = ts;
    state.idx = Math.min(state.idx + 1, state.timeline.length - 1);
    emit('frame-changed', { idx: state.idx, frame: currentFrame() });
    if (state.idx >= state.timeline.length - 1) {
      pause();
      return;
    }
  }
  requestAnimationFrame(playLoop);
}

export function jumpToEvent(evt) {
  setIdx(evt.frame);
  if (evt.node_id != null) selectNode(evt.node_id);
}

// Derived: every hazard-grade edge seen across the whole loaded timeline,
// with the frame index it occurred at -- backs the Hazard Timeline page and
// the "jump to hazard" timeline control, without re-deriving it per render.
export function allHazardEvents() {
  const out = [];
  const seenByNode = new Map();
  state.timeline.forEach((rec, idx) => {
    rec.edges.forEach((e) => {
      const sev = hazardSeverity(e);
      if (sev === 'safe' || sev === 'advisory') return;
      const prevSev = seenByNode.get(e.node_id);
      if (prevSev === sev) return; // only record the onset of a severity level, not every frame it holds
      seenByNode.set(e.node_id, sev);
      out.push({ frame: idx, t: rec.t, node_id: e.node_id, concept: e.concept, severity: sev, edge: e });
    });
  });
  return out;
}

// --- Spoken-directive audio playback ------------------------------------
// Real SeamlessM4T v2 clips exist per hazard edge (edge.audio, generated by
// generate_dashboard_data.py) but nothing ever played them back. Wired here,
// once, centrally -- not inside HazardPanel/HudPreview -- so exactly one
// clip plays per directive transition no matter how many components are
// currently showing the same edge, instead of each one triggering its own.
const _directiveAudio = typeof Audio !== 'undefined' ? new Audio() : null;
let _lastPlayedAudioKey = null;
let _playingSeverity = null; // severity tier of the clip currently in _directiveAudio
let _audioArmed = false; // skip the frame-changed that loadSession() itself fires -- no autoplay-on-load surprise

const SEVERITY_RANK = { advisory: 0, hazard: 1, critical: 2 };

export function setAudioMuted(muted) {
  state.audioMuted = muted;
  if (_directiveAudio) _directiveAudio.muted = muted;
  emit('audio-muted-changed', { muted });
}

on('session-loaded', () => {
  _lastPlayedAudioKey = null;
  _playingSeverity = null;
  _audioArmed = false;
});

on('frame-changed', ({ frame }) => {
  if (!_audioArmed) { _audioArmed = true; return; }
  if (!_directiveAudio || !frame) return;
  const edge = mostUrgentEdge(frame.edges);
  if (!edge || !edge.audio) return;
  const key = `${state.session}:${edge.audio}`;
  if (key === _lastPlayedAudioKey) return; // same directive still active -- don't restart it every frame

  // When two hazards have near-tied TTCs, which one mostUrgentEdge() returns
  // can flip from one frame to the next on ordinary numeric noise (not a
  // real change in the world). Without a guard, each flip reassigns
  // _directiveAudio.src and cuts the current clip off mid-sentence, which is
  // what "audio out of sync" actually was -- not a timing desync, but the
  // directive being interrupted and restarted by its own jitter. Only a
  // genuine escalation (advisory -> hazard -> critical) is allowed to
  // interrupt a clip that's still playing; a same-or-lower-severity flicker
  // waits for it to finish.
  const sev = hazardSeverity(edge);
  const stillPlaying = !_directiveAudio.paused && !_directiveAudio.ended;
  if (stillPlaying && SEVERITY_RANK[sev] <= SEVERITY_RANK[_playingSeverity]) return;

  _lastPlayedAudioKey = key;
  _playingSeverity = sev;
  _directiveAudio.muted = state.audioMuted;
  _directiveAudio.src = `/data/${state.session}/audio/${edge.audio}`;
  _directiveAudio.play().catch(() => {}); // blocked-autoplay (no user gesture yet) fails silently, not an error
});
