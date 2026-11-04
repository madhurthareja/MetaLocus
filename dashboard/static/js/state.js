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
  // Default to half real-time: the real device audience for this system
  // (a visually impaired walker) hears directives, doesn't read a data
  // feed, and full walking-pace playback compresses real-world gaps
  // between cues tighter than a first-time viewer can actually follow.
  speed: 0.5,
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

// Severity and urgency come from the pipeline (edge.severity, written by
// generate_dashboard_data.py via locus.actuation.directives.severity, which
// also applies the closest-approach gate: a closing object that will pass
// well clear is "advisory", not a hazard). Sessions generated before that
// field existed fall back to the TTC-only rule.
const PIPELINE_SEVERITY = { critical: 'critical', approaching: 'hazard', advisory: 'advisory' };
const SEVERITY_ORDER = { safe: -1, advisory: 0, hazard: 1, critical: 2 };

export function hazardSeverity(edge) {
  if (!edge) return 'safe';
  if (edge.severity) return PIPELINE_SEVERITY[edge.severity];
  if (edge.ttc_s == null) return 'advisory';
  if (edge.ttc_s < CRITICAL_TTC_S) return 'critical';
  if (edge.ttc_s < HAZARD_TTC_S) return 'hazard';
  return 'advisory';
}

// Same rule as locus.actuation.speech_policy.most_urgent_edge: highest
// severity first; among closing edges the soonest, among advisory ones the
// nearest.
export function mostUrgentEdge(edges) {
  if (!edges || !edges.length) return null;
  const key = (e) => {
    const sev = hazardSeverity(e);
    return [-SEVERITY_ORDER[sev], sev === 'advisory' ? Infinity : e.ttc_s, e.range_m];
  };
  return edges.reduce((a, b) => {
    const ka = key(a), kb = key(b);
    for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i]) return ka[i] < kb[i] ? a : b;
    return a;
  });
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
  // playLoop only emits frame-changed when idx actually advances, so the
  // frame already on screen when Play is pressed (frame 0 on a fresh
  // session, almost always) would otherwise never be (re-)announced --
  // the one time it was, from loadSession()'s own load-time emit, is
  // deliberately not treated as audio-eligible (see the session-loaded
  // handler below), since that fires before any user gesture. This one
  // does: it's the direct result of the Play click, so it's exactly the
  // moment a directive sitting on the current frame should become
  // audible, not one to suppress.
  emit('frame-changed', { idx: state.idx, frame: currentFrame() });
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
let _audioArmed = false; // disarmed only for the single synchronous frame-changed loadSession() itself fires

const SEVERITY_RANK = { advisory: 0, hazard: 1, critical: 2 };

export function setAudioMuted(muted) {
  state.audioMuted = muted;
  if (_directiveAudio) _directiveAudio.muted = muted;
  emit('audio-muted-changed', { muted });
}

on('session-loaded', () => {
  _lastPlayedAudioKey = null;
  _playingSeverity = null;
  // Disarmed only long enough to skip loadSession()'s own synchronous
  // frame-changed (so picking a session doesn't autoplay its clip before
  // Play is pressed) -- not "whichever frame-changed happens to arrive
  // first". That used to be the same thing, because this one fires
  // synchronously right after session-loaded and nothing else could beat
  // it to the listener. But most sessions' very first spoken cue lands
  // at t=0 (the directive pipeline evaluates frame 0 same as any other),
  // so the old "arm on first use" logic was silently discarding frame 0's
  // audio on every fresh load/play, forever -- the first thing the viewer
  // actually heard was already the SECOND cue, several seconds into a
  // video whose on-screen content (and the HUD/timeline reading it) was
  // otherwise exactly in sync. That one-cue lag is what read as "audio
  // and video are not synchronised." A microtask re-arms before control
  // returns to the event loop -- i.e. before Play, a scrub, or the
  // animation-frame loop can possibly produce the next real event -- so
  // exactly the one synchronous call gets skipped, never a real one.
  _audioArmed = false;
  queueMicrotask(() => { _audioArmed = true; });
});

on('frame-changed', ({ frame }) => {
  if (!_audioArmed) return;
  if (!_directiveAudio || !frame) return;
  // Only the edge the pipeline elected to speak carries a clip (see
  // locus.actuation.speech_policy) -- play that one, not whatever is most
  // urgent on screen, which may be something already announced.
  const edge = frame.edges.find((e) => e.audio);
  if (!edge) return;
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
