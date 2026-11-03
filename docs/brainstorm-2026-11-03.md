# Brainstorm: 2026-11-03: feasibility of five design changes

Status: brainstorm only. No repo code was changed. Every number below was
measured today, using throwaway scripts outside the repo (listed in the
appendix) against the shipped `dashboard/data/sanpo_edgetam_live_demo` run
(session `-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG`, 539 frames) and the cached raw
SANPO pose files.

Deadline context: Nov 4, 9:00 PM ET, about 1.5 working days away.

---

## TL;DR

While checking the premises behind the five ideas, I found two existing
bugs. Each one matters more than any of the five ideas, and all five sit
downstream of them:

1. **Left/right is currently a coin flip.** On frames where the wearer is
   looking roughly where they're walking, the shipped demo's left/right call
   matches the side of the image the object is actually on **55%** of the
   time (326/595). With the corrected convention it matches **99%**
   (588/595). The cause is not the up-axis ambiguity. SANPO's raw poses are
   *already* in the OpenCV camera convention with a Z-up world, so
   `_R_SANPO_LOCAL_FROM_OPENCV` is wrong outright. `frames.py` also assumes
   the world's vertical axis is Y.
2. **About 94% of critical alerts are velocity noise.** In the shipped run,
   342 of 539 frames have a critical edge (TTC < 2 s). Static objects carry
   a median estimated speed of **13 m/s**. Replaying with object velocity
   forced to zero (a static world) leaves **18 critical frames in 3
   episodes**.

**Recommendation:** spend the time fixing these two, then do #2 (as
tier-transition gating) and the CPA form of #3 for static objects. Both are
cheap and live in the same selection function. **Drop #1, #4 and #5.**
Reasons are in each section below.

---

## 0. The left/right question (answered first, as asked)

### Would a chest-camera lateral-position test settle it? No.

- The head and chest cameras are stacked vertically on the same person, with
  almost no lateral offset between them, so a lateral comparison has nothing
  to measure.
- The chest poses come from the same kind of pose file, so they share any
  convention error. They can't serve as an independent reference for
  *convention*, only for *body heading*.
- The existing chest comparison
  (`test_velocity_derived_heading_matches_chest_camera_measured_heading`)
  measures yaw as `atan2(v[0], v[2])`, which assumes Y is up. SANPO's world
  is Z-up, and the wearer walks mostly along **+Y**. That yaw throws away
  the dominant motion component and compares the leftover X/Z components of
  two vectors. Under the old convention, both leftovers happened to point
  toward −X. That is how "forward = local −X, median 22°" passed. The match
  was an artifact of projecting onto the wrong plane.

### Two checks that do settle it (both run today, about 20 minutes each)

**(a) Multi-view depth reprojection over every candidate convention.**
For 10 frame pairs 0.5–1 s apart, I lifted about 4,000 depth pixels from
frame i, transformed them to world coordinates and back into frame j, and
compared the result against frame j's measured depth. The search covered
48×48 signed axis permutations: the camera-side correction × a world-side
correction.

| Convention | Median relative depth error |
|---|---|
| Raw SANPO poses, no correction at all (OpenCV camera, world as given) | **16.2%** |
| Current repo `_R_SANPO_LOCAL_FROM_OPENCV` | **44.7%** |
| Pre-"fix" repo matrix (before the right-row sign flip) | 48.2% |
| Median over all 2,304 candidates (≈ chance) | 44.3% |
| Translation zeroed (control: proves translation carries signal) | 28.3% |

The current matrix scores no better than chance. It also has
**determinant −1**, which makes it a reflection rather than a rotation;
the pre-fix matrix had +1. The best-scoring candidate (14.0%) is "no
camera correction" plus a Z flip on positions. That flip only touches the
small vertical translation (Z spans 1.1 m), so I'd treat it as noise, not
a finding. It doesn't affect left/right.

With raw poses, the axes are exactly what you'd expect for a person
walking:

- camera forward ≈ world **+Y** (mean 0.894), matching the walking
  direction over the first 10 s (+Y component 0.984)
- camera down ≈ world **−Z** (−0.944), so **Z is up**, consistent with Z
  being the bounded translation axis
- camera right ≈ world **+X**, which makes a proper, right-handed frame

**(b) Image-side check, the one that matches what the user hears.** I
picked frames where the camera faces within 20° of the walking direction,
and objects more than 10° off image center. On those, is the object
reported on the same side as the half of the image it's in? Shipped:
**55%**. Corrected: **99%**.

### Why the earlier "fix" looked right

`frames.py` builds the body frame as `right = cross(_WORLD_UP, forward)`
with `_WORLD_UP = [0, 1, 0]`. On SANPO, forward ≈ +Y, so this cross
product is built almost entirely from the small, noisy X/Z components of
the wearer's velocity. The resulting "right" axis is close to vertical.
The van on frame 8 flipped sides because of a reflection, not a real fix.
The 72.9% / 27.1% mirror in
`test_camera_right_maps_to_positive_body_frame_bearing` is what flipping
the sign of *anything* would produce. It doesn't validate handedness.

A naming trap worth knowing: in the synthetic world, `R_world_cam` is
identity at yaw 0 (`synthetic.py:84`), so world axes equal OpenCV camera
axes and **world +Y points down**. `_WORLD_UP = [0,1,0]` is therefore
really *gravity-down*, and `cross(down, forward)` correctly gives right.
For SANPO, the value to pass is `[0, 0, -1]`. With that one change, the
existing `body_rotation_from_velocity` gives the 99% result above, with no
new math needed.

**AEA has the same problem.** Camera +X maps to world −Z (mean −0.70; the
Aria RGB sensor is mounted rotated), and Z is the bounded translation axis
(0.67 m range vs 11.4 / 9.9 m). So AEA is Z-up too. The ablation
(`generate_ablation_results.py`) is synthetic-only, so its published
numbers are unaffected.

### A second geometry problem found on the way: half the frames have no heading

- **271 of 539 frames have exactly zero wearer velocity.** On those,
  `body_rotation_from_velocity` falls back to forward = world +Z, which on
  SANPO points straight up. Bearings on those frames mean nothing.
- Two causes:
  - The wearer walks at about 1.17 m/s for roughly 13 s and then stands
    still for most of the rest. Only 198 of 539 frames are walking, using
    1 s-smoothed speed > 0.3 m/s.
  - The loader replaces the 160 `NOT_READY` pose rows (30%) with the
    nearest READY pose. Consecutive frames then have duplicate poses, and
    the velocity between them is zero even mid-walk.
- `frames.py`'s docstring says the heading comes from "smoothed"
  velocity, but `generate_dashboard_data.py:298-302` uses a raw one-frame
  finite difference.
- The fix: a 1 s causal velocity window, and hold the last heading when
  speed drops below about 0.3 m/s.

**Re-measured heading error under the correct convention:** velocity
heading vs. chest-camera heading is **14.7° median, 23.8° p90** (n=62
walking frames). The ~22° figure in `CLAUDE.md`, the loader docstring and
`system_facts.json` came from the flawed comparison and should be
replaced. Head camera vs. velocity heading is 9.9° median.

### Fix A: effort

| Step | Effort |
|---|---|
| Loader: drop the correction matrix (use identity) | 15 min |
| Thread a `world_down` parameter through `world_to_body` → `compute_edges` → `SpatialMemoryGraph` (`body_rotation_from_velocity` already accepts it). SANPO/AEA pass `[0,0,-1]`; synthetic keeps `[0,1,0]` | 1 h |
| `edges.py:36` `elevation_delta_m = relative[1]` → use the down axis | 10 min |
| 1 s wearer-velocity window + hold heading when stopped (`generate_dashboard_data.py`, `run_real_sequence.py`, `eval/conditions.py`) | 45 min |
| Rewrite the two SANPO tests: heading test yaw in the XY plane; replace the camera-right test with the image-side test above | 1 h |
| Correct the docs claims ("forward = local −X", "22°", "right-axis fixed") | 30 min |
| Regenerate sessions: about 1 h per EdgeTAM session, judging by log timestamps; unattended | 1–3 h wall clock |

Total: about 3.5 h of hands-on work, plus regeneration time.

---

## 0b. Object velocity is mostly noise

- In the corrected replay, static objects' range jumps by a median of
  **0.14 m per frame**, p90 **0.70 m**, against at most 0.08 m from walking.
  At 15 fps, that's about 2 m/s (median) to 10 m/s (p90) of fake motion
  per raw sample. The `VELOCITY_SMOOTHING_ALPHA = 0.3` smoothing in
  `graph.py` can't absorb that.
- Velocity is also **frozen** while a node goes unobserved. A "person"
  held 36.9 m/s for about 2 s, and a "truck" peaked at 112 m/s, producing
  TTCs of 0.02–0.2 s at 6 m range.
- The jumps come from EdgeTAM masks drifting and from lifting noise. That
  root cause can't be fixed in 1.5 days, but its effect on velocity can.

| Alert set | Critical frames | Episodes |
|---|---|---|
| Shipped | 342 / 539 | (not counted) |
| Corrected convention, velocities as estimated | 298 | 42 |
| Corrected convention, static world (object v = 0) | **18** | **3** (pole 0.7 s, truck 5.1 s, pole 13.7 s) |

22 of the 25 clips the shipped run actually spoke were "critical", so in
practice the demo's spoken output is mostly driven by this noise.

**Fix B: effort.**

- **Class prior** (30 min): pole, tree, hydrant, bin, lamppost, sign and
  traffic light get v = 0. Honest and cheap.
- **Stop freezing velocity** (30 min): decay it to zero when a node hasn't
  been observed for more than about 0.5 s.
- **Optional, 1–2 h:** fit velocity as a least-squares slope over ≥1 s of
  observations with a deadband, for the concepts that can move.

---

## 1. Ego-speed substitution

**Where ego speed is derived:**
- `generate_dashboard_data.py:298-302`: one-frame finite difference of
  `pose.t_world_cam`
- `run_real_sequence.py:~163` and `eval/conditions.py:66-81`: same pattern
- `config/default.yaml` `kinematics.walking_speed_mps: 1.4` is **unused**
  by the real pipeline. `synthetic.py` has its own constant.

**Where it's consumed:**
- `graph.edges()` → `compute_edges()` (`edges.py:27-30`) uses it in two
  places: `world_to_body` (direction) and `closing_speed` (magnitude).

**Does `frames.py` use velocity only for direction?** Yes, confirmed.
`frames.py:39-40` normalizes the velocity, and magnitude only matters
through the `> 1e-6` fallback check. So substituting a different speed
won't corrupt the direction. It can't fix the direction either: on 50% of
this session's frames there is no direction to scale, because the speed is
zero. Any substitution has to be gated on actually walking.

**Is closing speed separable into ego and object parts?** Yes, cleanly.
`kinematics.py:34-35` computes `−(v_obj − v_wearer)·los`, which is linear,
so it splits as `v_wearer·los − v_obj·los`. Splitting it is a two-line
change.

**Could it be two columns in the ablation?** Not as things stand. The
ablation is synthetic-only and reports answer rate, bearing error and range
error. Its own docstring lists TTC and hazard metrics as `not_implemented`,
so there is no hazard column to split by pace. You'd have to build the
hazard metric first, which is 3–5 h.

**What breaks or becomes misleading:**

- **For static objects, it's just a threshold change.** TTC = r / (v·cosθ),
  so "TTC at the slow pace < 2 s" is identical to "TTC at the recorded pace
  < 2·v_slow/v_recorded". It doesn't model anything that changing the
  threshold wouldn't.
- **It points the wrong way.** A slower walker gets *fewer* TTC alerts,
  but the BLV argument is about needing *more* lead time. If the motivation
  is BLV realism, raise the threshold (or switch to a distance-based lead)
  and say so directly.
- **The timeline stops being consistent.** Recorded positions still advance
  at the sighted pace, so objects close in faster than the substituted
  TTC claims. The sequence of TTCs is not a slow walk.
- **Measured:**
  - With the current noisy velocities: 298 → 298 critical frames at
    1.0 m/s. No effect, because noise dominates.
  - After fix B (static world): 18 → **2** frames at 1.0 m/s. The result is
    extremely sensitive to the chosen pace, which is exactly the
    threshold-knob behavior described above.
- If you do quote a BLV walking pace anywhere, cite a source. I haven't
  verified a figure.

**Effort:** 1 h for the substitution alone; 3–5 h including a hazard metric
for the ablation.

**Verdict: don't.** Push back on this one.

---

## 2. Novelty-gated alerting

**Does the graph already carry what's needed?** Partly. `Node` has
`first_seen`, `last_seen`, `observation_count`, `confidence` and `visible`
(`types.py:55-65`). What's missing is a per-node record of what was last
announced: time, tier, clock bucket and range bucket. That's
actuation-layer state and shouldn't live in `Node`.

Novelty is also only as reliable as node identity. With position noise at a
p90 of 0.7 m per frame against a 1.0 m association gate, a node that splits
looks "new" again.

**Where it belongs:** in the directive-selection block in
`generate_dashboard_data.py:342-357`, with a mirror in `state.js`'s
playback logic (around line 215). Not in the graph, and not in the TTC
layer.

**How it interacts with the cooldown:** they compose; neither replaces the
other.
- **Novelty** decides *which* edges are eligible to speak.
- **Cooldown** decides *when* the channel is free.
- The existing critical-escalation bypass must bypass the novelty gate too.

**Push back on the definition.** "Stable for 10 s isn't news" is wrong for
an object you're walking toward, because your relationship to it keeps
changing. Gate on a *change in announced state*:
- a new node
- a tier change (info → advisory → critical)
- a clock-position change of about 2 hours
- re-entry after a long absence

Don't gate on node age. With that definition, a tree you pass is spoken
once, and a pole you're closing on speaks again when its tier changes.

**Measured on the shipped run:** 25 spoken events across 7 nodes, the
worst being tree ×6, pole ×5 and car ×5. Gating on tier or 2-hour changes
keeps 19. That's a small effect today, because the cooldown already does
most of the work and the noise makes states change constantly. Its value
shows up only after fixes A and B.

**Effort:** 1.5–2 h, covering Python, the JS mirror and a test.

**Verdict: do it, after A and B.**

---

## 3. Bearing stability (CBDR) as the alert criterion

**Is enough bearing history kept?** No. Edges are recomputed every frame,
and `Node` stores only the current position and velocity. Online, you'd
need a per-node ring buffer of about 45 entries (3 s × 15 fps) holding
time, horizontal relative position and range. Offline, `timeline.json`
already has it.

**The crux: does the heading error cancel?** The question doesn't need
answering. Compute the line-of-sight azimuth in the **world** frame
(`atan2` of the horizontal relative position) instead of body bearing.
CBDR is about the line of sight rotating in an inertial frame, so the
heading estimate never enters. Body-bearing change would only cancel the
heading bias if that bias stayed constant over the window, and
velocity-derived heading wobbles with gait.

The heading error is also now 14.7°, not 22° (see section 0).

**Noise vs. signal, measured (corrected replay):**
- **Noise:** for static objects within 8 m while walking, the world
  line-of-sight azimuth changes by a median of **2.0° (p90 10.0°)** per
  1/3 s (n=152).
- **Signal:** an object 1 m off your path, 4 m away, at 1.17 m/s rotates
  about **4°/s**, which is about 8° over 2 s.

So noise per sample is the same order as the 2 s signal. An endpoint
difference won't work; you need a slope fit over the window. Body-frame
bearing jitter with a 1 s-smoothed heading is fine: median 0.8°, p90 4.1°
frame to frame.

**How many current alerts would survive?** Measured on the corrected
replay's 298 critical alerts:

| Window | Alerts with full history | Survive Δaz<3° & range shrinking | <5° | <10° |
|---|---|---|---|---|
| 2 s | 123 | 23 | 27 | 73 |
| 3 s | 113 | 4 | 7 | 32 |

Caveat: that alert set is itself mostly velocity noise (section 0b), so
this measures a noise filter applied to noise. It shows CBDR removes about
80% of alerts, but not whether it keeps the right ones. After fix B, there
are only 3 episodes left, too few to judge.

**Push back on the formulation.** CBDR is a stand-in that sailors use
because they have bearings but not range rate. We have range *and* 3D
relative velocity, so we can compute the closest point of approach (CPA)
directly: `miss = |r × v_rel| / |v_rel|`. It's the same physics with no
history needed. The catch is that CPA uses velocity, the noisy quantity.
So the right split is:
- **Static objects:** after fix B's class prior, v_rel = −v_wearer, which
  is well estimated. CPA is then exact and costs about 1 h, including an
  `Edge` field and the selection filter. It directly kills the "pole off to
  the side" alerts. Measured: CPA < 1 m keeps all 3 static-world episodes,
  on 3 of their 18 frames.
- **Moving objects:** CBDR on position history is more robust than CPA on
  noisy velocity. It costs 3–4 h with the ring buffer, a robust slope fit
  and dashboard display.

**Verdict:** do CPA for static objects. CBDR for moving objects only if
time remains; I don't expect it to.

---

## 4. Three-register output policy

**How much already exists:** less than it appears.
`actuation/audio.py:render_cue` (stereo pan plus TTC pulse) exists, but
**nothing calls it**, neither the pipeline nor the dashboard. The
dashboard's "tone: panned right" is text only (`components.js:215, 274-275`);
no tone is ever played. Playback is a single `HTMLAudioElement`
(`state.js:195`) driven by one `edge.audio` field. It's one channel from
end to end.

**Where tier selection would live:** in the same selection block as the
cooldown and novelty gate. It would emit a per-frame record like
`cue: {register, node_id, …}`. Earcons are best synthesized client-side
with Web Audio (`StereoPannerNode` plus an oscillator), with no files.

**Can the timeline show three channels?** No, it's built around one. Adding
lanes means real work in `components.js`'s timeline.

**What's hard:**
- **Orientation "on request" has no input path.** `src/locus/query/` is
  empty. "At a pause" is easy to detect, though; this session has a long
  stop.
- **The narrator would need repositioning.** The imperative register's ≤6
  words already matches `describe()`, so yesterday's ≤12-word narration
  would move to the orientation register.
- **Learnable earcon design is a UX research problem.** It can't be
  validated in 1.5 days.

**Effort:** 6–10 h (pipeline 2 h, earcons 2–3 h, timeline lanes 2–3 h,
orientation summary 2 h).

**Verdict: doesn't fit alongside A and B. Drop it.**

---

## 5. Scoping alerts to the cane's blind spots

**Is the up-axis question blocking?** Not anymore; section 0 settles it.
Height above ground is still missing, though:
- `Node.p_world` is the mask centroid, and the extent lives only in a side
  dict in `generate_dashboard_data.py`.
- "Above knee height" needs the object's *bottom* height, which means the
  per-mask minimum world-Z from `lifting.py`.
- It also needs a ground reference. Either assume a fixed head height
  (crude) or fit a per-frame ground plane on depth (about 2 h).

**The vocabulary fights the idea.** Everything Spark proposed for this
session (pole, hydrant, tree, van, truck, bin, person, car, traffic light,
taxi, lamppost) touches the ground and is cane-detectable, apart from
things that move. Scoping would silence almost every static alert. Real
blind-spot objects (branches, signs, awnings, truck mirrors) aren't in the
vocabulary, and I haven't verified that SANPO's sessions contain them.

**"Fast silent approach":** there's no audio input to judge "silent", and
"fast" depends on velocity, which is bug B.

**Is "entered from outside the previous FoV" derivable?** Mechanically,
yes: project the new node into the previous frame with
`intrinsics.in_frustum`, as `graph.predict_reentry` already does. But the
head camera's FoV sweeps with every head turn, so static objects "enter
the FoV" constantly. You'd have to AND it with world-frame motion, which
is bug B again.

**Effort:** 6–8 h plus a vocabulary change, with no evaluation data.

**Verdict: doesn't fit. Drop it.** The framing ("we don't duplicate the
cane") is the strongest product story of the five, but there's no
time to build or validate it.

---

## Recommended order for the remaining time

1. **Fix A** (convention, world-down, heading hold): about 3.5 h.
2. **Fix B** (velocity class prior, no frozen velocity): about 1 h.
   Regenerate once after both, since narration and TTS clips regenerate
   too.
3. **#2** tier-transition gating plus **#3-CPA** for static objects: 2–3 h.
4. Re-measure on the regenerated run and update the docs:
   - critical frames
   - spoken clip count
   - image-side left/right agreement (target ≈99%)
   - heading error (14.7°)
5. Drop #1, #4 and #5.

Risk: regenerating the day before submission changes what the demo shows.
Keep the current session directory until the new one has been checked
frame by frame.

---

## Appendix: how this was measured

Scratch scripts, not in the repo:
`/tmp/claude-1001/-home-madhur-Desktop-hackathon/42eef65f-0ebd-4d2e-9fe9-2e544893c291/scratchpad/`
(`convention_search.py`, `score_report.py`, `inspect_best.py`, `replay.py`,
`replay2.py`). Run with the `hackathon` conda env.

- **Reprojection search:** uses raw quaternions and positions from
  `fixed_camera_poses.csv` and the cached depth maps. READY frames only.
  Frame pairs (20,28), (20,35), (45,53), (45,60), (75,83), (75,90),
  (100,108), (100,115), (120,128), (120,135).
- **Replay:** inverts the recorded `camera_R_world_cam` to recover each
  observation's camera-frame point, re-lifts it with the raw quaternion
  (`R_true = R_shipped @ M_shipped.T`), and replays the repo's own
  `SpatialMemoryGraph`. Edges are computed with
  `body_rotation_from_velocity(heading, up_world=[0,0,-1])`, a 1 s causal
  wearer velocity, and the heading held when speed < 0.3 m/s.
- **Replay limitation:** it reuses the shipped run's detections and the
  node associations made under the wrong convention. It approximates a full
  re-run; it is not a substitute for one.
