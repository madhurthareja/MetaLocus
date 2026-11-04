# Project context for Claude Code

Locus -- Meta Global AI Developer Hackathon, Indie track, team of 2. Full
strategic plan: [PROJECT_PLAN.md](PROJECT_PLAN.md) (still the right
reference for the pitch/benchmark/model-stack rationale; its module-path
examples predate the `src/locus/` restructure below -- trust this file and
the code over those specific paths).

**Note:** this file lives in `docs/` rather than the repo root, so it is
not auto-loaded as Claude Code project memory. Move or symlink it to the
repo root if auto-load is wanted.

## Timeline
Kickoff Oct 26, submission due **Nov 4, 9:00 PM ET**.

## Hardware
NVIDIA DGX Spark (GB10, aarch64/Grace-Blackwell), 128GB unified memory,
CUDA 13.0 toolkit, torch 2.10.0+cu128 (works despite GB10's compute
capability 12.1 being just above that build's stated max of 12.0 --
verified with a real CUDA matmul, not assumed). Memory capacity is not the
constraint; per-concept SAM 3.1 session throughput (each reruns the
Perception Encoder) was the untested unknown -- now measured for real by
`scripts/smoke_test_sam31.py` (no longer a stub): 1/3/6 concurrent concept
sessions over the same real frame window took 35.3s/77.1s/144.4s (~35s,
~26s, ~24s per session respectively) -- `prepare()` runs sessions strictly
sequentially, so elapsed scales with concept count, not GPU contention.
Results in `dashboard/data/sam31_smoke_test.json`.

## Naming: Locus, not hazard_agent
The project was restructured mid-build around a more detailed, independently-
written spec (internally referred to as "chat.md") that used the name
`locus` and a specific module layout. That restructure **kept** the
hazard/TTC/no-GATv2 decisions from the original plan, and **adopted**
Locus's module layout, three-reference-frame design, and A/B/C/D ablation
harness. One explicit deviation from that spec: this repo stayed at its
original path (`/home/madhur/Desktop/hackathon`, which already had git
history and a pushed GitHub remote) rather than moving to a fresh
`~/projects/locus` -- confirmed with the user, not a silent judgment call.

**Dataset status has since changed from that restructure: AEA was
initially kept as "the real benchmark data source," but that's now
superseded.** SANPO is the primary, scored dataset (see
`PROJECT_PLAN.md` §1-3 for the full argument: AEA's indoor clutter
required reframing household obstacles as "hazard proxies," which doesn't
hold up, while SANPO is purpose-built for outdoor BLV navigation and
already powers Google's Project Guideline in production -- no reframing
needed). AEA is kept as a secondary, cross-dataset validation that the
identical pipeline generalizes to a second sensor/depth modality, not as
the hazard-narrative dataset.

## Key architecture decisions (see PROJECT_PLAN.md for full rationale)
- **SAM 3.1 Object Multiplex, required by the hackathon's Meta Model API
  tooling.** Native path is one session per concept (no multi-concept
  single pass), 4-6 live concept slots. `src/locus/perception/sam3_native.py`
  documents this constraint explicitly -- it's a real correction against
  chat.md's own (incorrect) assumption that SAM 3 supports multi-prompt
  sessions.
- **SANPO primary, AEA secondary, neither is EgoObjects.** SANPO (outdoor)
  is the scored dataset -- purpose-built for BLV navigation, already
  powers Project Guideline, real dense depth and real 6-DoF pose from two
  simultaneous cameras. AEA (indoor) runs the same unchanged pipeline as a
  cross-dataset generalization check. EgoObjects has neither real depth
  nor pose, for either role. Full argument in `PROJECT_PLAN.md` §1-3.
- **No GATv2 / learned GNN.** Time-to-collision is `distance / closing
  speed`, computed geometrically in `src/locus/geometry/kinematics.py`.
- **Open-loop trajectory, not closed-loop.** Mirrors what an on-device
  system would actually have at inference time; this is what
  `src/locus/data/aea_loader.py` parses.
- **Three reference frames** (`src/locus/geometry/frames.py`): F_ego
  (camera), F_body (torso/clock-face, +Z = direction of travel from
  smoothed world-frame velocity, independent of head orientation),
  F_world. The invariant this all depends on -- a static object's
  body-frame bearing stays constant under pure camera rotation -- is
  tested directly in `tests/test_geometry.py`; if that test ever fails,
  nothing built on top of it is trustworthy.
- **locus.data.synthetic is a unit-test harness, not a benchmark
  substitute.** SANPO (and AEA, secondarily) are the real benchmark data
  sources. Synthetic scenarios exist because they're the only data source
  with exact ground truth for every quantity (visibility, bearing, TTC)
  with no weights/keys/network needed -- see `src/locus/eval/questions.py`
  and `tests/test_eval.py` for
  how the same conditions run against both.

## What's real vs. stubbed, as of this writing
Verified and tested (21 tests passing, `pytest tests/ -v`):
- Geometry (3 frames, lifting, kinematics/TTC) -- `tests/test_geometry.py`
- Persistent memory graph (decay, eviction, re-association) --
  `tests/test_memory.py`
- A/B/C/D eval harness, run against synthetic scenarios, including a
  regression test that the harness still demonstrates the actual thesis
  (persistent memory beats stateless captioning on out-of-view questions)
  -- `tests/test_eval.py`
- Audio actuation (stereo pan toward bearing, TTC-pulse rate, ≤6-word
  spoken directives) -- `tests/test_actuation.py`
- Real AEA sequence loading: open-loop pose + fisheye-to-pinhole RGB
  undistortion, verified against actual rendered frames (not just unit
  tests) and run successfully across all 8 downloaded sequences --
  `tests/test_aea_loader.py` (skips cleanly if the dataset isn't present
  on a given machine)

Real and verified since the above was first written:
- **Real AEA sparse depth** (`load_point_cloud` + `project_sparse_depth` in
  `aea_loader.py`): projects the semi-dense SLAM point cloud into each
  frame's real pose as a per-pixel z-buffer (nearest point wins, same
  occlusion handling as `locus.data.synthetic`'s farthest-first painting).
  ~18% pixel coverage on the sequence tested -- sparse, but that's exactly
  the shape `lift()`/`lift_from_box()` already expect.
- **SAM 3.1 native segmenter** (`src/locus/perception/sam3_native.py`) --
  real session wiring against the actual downloaded checkpoint, run
  successfully on real AEA frames. Architectural note: SAM 3.1's video API
  is batch (`start_session`/`add_prompt`/`propagate_in_video` over a
  materialized frame directory), not streaming like `MockSegmenter`, so
  this class has a `prepare()` step that does the real model work up
  front and `step()` just replays the precomputed per-frame result.
- **`scripts/run_real_sequence.py`** -- the complete real thread, one-shot:
  real pose + real undistorted RGB + real sparse depth + real SAM 3.1
  masks + persistent memory graph + TTC + spoken directives, run
  end-to-end on `loc5_script4_seq6_rec1`. Mask quality verified visually
  (not just "it ran") -- correctly segments both near and far chairs, the
  couch, and the table with clean instance-level boundaries. This is the
  first real validation this project has of SAM 3.1 output quality, not
  just throughput.
- **Three real environment bugs found and fixed getting there** (GB10 is a
  genuinely new platform; this is exactly the "day-1 risk" PROJECT_PLAN.md
  §8/§11 flagged, just found a few days later than planned):
  1. `pkg_resources` import failure -- recent `setuptools` dropped the
     vendored module; pinned to `setuptools<81`.
  2. Missing `einops`/`pycocotools` -- transitively imported by `sam3`'s
     training-data code path even for pure inference; installed via the
     repo's own `dev` extras (`pip install -e ".[dev]"`).
  3. **NVRTC couldn't JIT-compile a reduction kernel for GB10's compute
     capability 12.1** (`invalid value for --gpu-architecture`) --
     pre-compiled kernels (matmul etc.) work fine on 12.1 despite torch's
     stated max of 12.0, but the Jiterator's on-the-fly kernel compile
     path is pickier about the exact arch string. Fixed by upgrading just
     `nvidia-cuda-nvrtc-cu12` to the latest available (12.9.86) --
     narrow, doesn't touch the rest of torch's ABI-sensitive binary.
  4. (Not a bug, a real API-version mismatch in the vendored `sam3`
     checkout) `Sam3BasePredictor.start_session` unconditionally passes
     `offload_state_to_cpu` to `model.init_state`, which this checkout's
     multiplex model doesn't accept. Worked around in our own code
     (`Sam31NativeSegmenter._start_session`) by calling `model.init_state`
     directly and registering the session the same way the base class
     does, rather than patching the vendored repo.
  5. `sam3`'s own `numpy<2,>=1.26` pin is now the **active** environment
     numpy (1.26.4, downgraded from 2.5.4 by installing `sam3[dev]`) --
     the previously-flagged conflict is no longer hypothetical. Full test
     suite re-verified green under 1.26.4; no actual breakage found
     despite `pip`'s dependency-conflict warnings for a few other
     packages (opencv-python-headless, scipy, contourpy, rerun-sdk all
     declare `numpy>=2` but still work at runtime).
  6. FA3 (`use_fa3=True`, SAM 3.1's default) needs the separate
     `flash_attn_interface` package, not installed; `Sam31NativeSegmenter`
     now defaults `use_fa3=False` (plain attention, some speed cost, no
     separate build needed).
- **A real duplicate-node bug, found on the real-data run and fixed**: SAM
  3.1's own tracker can emit two different instance_ids for the same
  physical object (confirmed visually -- two overlapping chair masks in a
  single real frame, lifted to 3D positions 0.29m apart). `SpatialMemoryGraph`
  only checked its association gate once, at node-creation time, so once two
  instance_ids landed on separate nodes they stayed separate even after
  later observations brought them within the gate of each other. Fixed in
  `memory/graph.py` (`_merge_duplicate_nodes`/`_merge_into`, called at the
  end of every `observe()`): re-checks all same-concept node pairs after
  each observation and merges any that have converged within the gate,
  keeping the more-recently-observed node's state and redirecting the
  merged-away node's instance_id mappings. Covered by
  `test_duplicate_instance_ids_merge_once_they_converge_spatially` in
  `tests/test_memory.py`. Verified against the real sequence that exposed
  it: node count there dropped from 11 to 10, the duplicate pair gone,
  the two genuinely-distinct nearby chairs (1.48m apart, outside the gate)
  correctly left alone.
- **The dashboard** (`dashboard/app.py` + `scripts/generate_dashboard_data.py`)
  is the real pipeline's actual front end now, superseding
  `scripts/run_real_sequence.py` as the way to produce and view a full
  real-data run: Spark concept proposal (re-asked at checkpoints through
  the clip, not just frame 0 -- a one-shot opening-frame proposal can never
  see a concept that enters later), segmentation, lifting, the persistent
  memory graph, TTC, directives, real TTS, muxed into per-session
  `frames.mp4`/`masks.mp4`/`timeline.json`/audio that a 5-page browser UI
  replays.
- **EdgeTAM live-segmentation backend**
  (`src/locus/perception/edgetam_segmenter.py`), added alongside SAM 3.1 as
  a second, genuinely-causal/streaming `Segmenter` (`--segmenter edgetam`
  on `generate_dashboard_data.py`), for the real-time path SAM 3.1's
  whole-clip batch design can't offer (see `sam3_native.py` for that
  constraint). Loaded via `transformers`' native EdgeTAM port
  (`yonigozlan/EdgeTAM-hf`, 56MB, model_type `edgetam_video`), which, like
  the rest of SAM 2's lineage, is point/box-prompted, not text-prompted --
  something still has to turn a Spark-proposed concept string into a box.
  That's `Sam31NativeSegmenter.ground_text()` (new): one cheap
  `add_prompt` call, no `propagate_in_video`, used as EdgeTAM's default
  grounder. Measured on real SANPO frames on GB10: ~10.5 FPS tracking 3
  simultaneous objects causally (vs. SAM 3.1's 0.17-0.68 FPS batch
  propagation) -- a real, large speed win, at a real, large accuracy cost
  on long clips (below).
  - **A real cost bug, found and fixed**: a concept that never grounds
    (Spark proposed it, nothing there) was retried every single frame
    forever -- 687 SAM 3.1 grounding calls for 10 concepts across one
    539-frame run, which would make this impractical for actual live use
    (paying for a full SAM 3.1 session every frame indefinitely for
    anything not yet in view). Fixed with `ground_retry_interval` (default
    15 frames) -- same run afterward: 30 calls.
  - **A real accuracy limitation, measured, not assumed**: EdgeTAM is
    causal with no reverse pass and no occlusion re-identification. On a
    full real 539-frame (36s) SANPO walk, tracked node count fell from 11
    live mid-clip to 2 by the end (vs. SAM 3.1's batch pass holding ~84
    steady), and only 13.8% of raw detections lifted to 3D (vs. SAM 3.1's
    ~93%). This is the actual EdgeTAM/SAM-2-lineage tradeoff, not a
    regression.
  - **Mitigation: periodic re-grounding** (`reground_interval`, default 45
    frames) -- every interval, each already-tracked concept gets one fresh
    SAM 3.1 box re-anchored to its *existing* object id (lands on a
    frame_idx the session has never seen, so it's automatically treated as
    a fresh conditioning frame, no special "reinit" path needed). A
    transient miss (object briefly occluded) leaves the existing track
    alone rather than dropping it. Same 539-frame run with this on: final
    node count 2 -> 9, lift rate 13.8% -> 45.0%, node count held near-peak
    (11 -> 9) almost to the end of the clip instead of collapsing partway
    through. Still short of SAM 3.1's batch-pass numbers (expected -- a
    periodic box re-anchor is a cheaper correction than full
    forward+reverse occlusion reassociation), but a large, real
    improvement over uncorrected causal drift.
- **Spoken-directive pacing** (`locus/actuation/speech_policy.py`, used by
  `generate_dashboard_data.py`): found and fixed a real UX problem, not a
  technical bug -- a directive+audio clip was being attached to *every*
  tracked hazard edge on *every* frame, so a brand-new clip would fire the
  instant the previous one finished playing the moment phrasing shifted by
  even one clock-position bucket. A blind listener hears this in real
  time, not as a data table; constant back-to-back audio with no gap to
  process it is unusable even when each individual cue is accurate. Now at
  most one edge is elected to speak per frame, gated by a cooldown since
  the last spoken cue (3.0s normal, 1.2s if still critical) -- except a
  genuine escalation into critical range, which always cuts in
  immediately. Measured effect on the same real 539-frame run: 252 unique
  spoken clips -> 23. **Then (2026-11-03) a novelty rule**: an object is
  announced once, and again only when something about it changes (severity
  goes up, or it moves 2+ clock hours while ahead -- not the ordinary drift
  of walking past it); routine mentions skip things behind the wearer. On
  the same session, replayed: 25 spoken cues -> 12. **Then (2026-11-04) a
  timeliness rule**: objects entering range together queued behind the 3s
  cooldown, so routine mentions arrived a median ~6s (up to ~24s) after the
  object appeared -- heard as "the voice is late". Routine mentions now
  only start within 10m (`ADVISORY_RANGE_M`), and one that can't be spoken
  within 2s of becoming news (`ADVISORY_MAX_WAIT_S`) is dropped rather than
  spoken late. Warnings (approaching/critical) are never dropped. Playback
  itself was measured at 44-93ms from trigger to audible. Dashboard default
  playback speed also dropped from 1x to 0.5x
  (`dashboard/static/js/state.js`) -- the real audience hears this in real
  time rather than reading a feed, and full walking-pace playback
  compresses real-world gaps between cues tighter than a first-time
  listener can actually follow.
- **False hazard alerts were ~94% velocity noise (fixed 2026-11-03).** On
  the shipped SANPO run, 342 of 539 frames carried a critical (TTC < 2s)
  alert. Three causes, three fixes, each measured on that session:
  1. Static objects picked up 1-13 m/s of "velocity" from per-frame
     lifting jitter, frozen in place once unobserved.
     `memory/motion_classes.py` sorts concepts into static (pole, tree,
     hydrant, ... -- velocity pinned to 0), movable (person, car, ... --
     per-class speed ceiling, 1s decay half-life, 4s eviction, no edge once
     unseen for 0.75s) and unknown (unchanged behaviour). 342 -> 43
     critical frames.
  2. EdgeTAM regularly hands one instance id a different physical object
     (one "truck" node re-attached to 19 trucks in 36s, its range
     alternating 6m/21m on consecutive frames), and the old velocity rode
     along onto the new object. A step faster than the class can move now
     starts a new track segment at zero velocity, and mover velocity is a
     least-squares fit over the current segment (<=1s, needs >=5
     observations spanning >=0.4s). 43 -> 21.
  3. What remained was walking *past* things -- poles a metre to the
     side, closing fast but never reached. Edges now carry the horizontal
     closest point of approach (`cpa_m`); a closing object only counts as
     a hazard if it will come within `CPA_GATE_M` = 0.75m
     (`actuation/directives.py`: ~0.3m shoulder half-width + ~0.45m for
     heading error). Every remaining pole alert had a 1.0m miss distance.
     21 -> 2 critical frames. Those objects are still mentioned, just not
     as hazards. Honest caveat: a gate this close to the measured 1.0m
     means the choice of 0.75 vs 1.0 matters, and position error near the
     gate can still flip one.
  4. The last two were parked vehicles beside the sidewalk whose lifted
     position slid ~1 m/s toward the wearer for exactly one frame as they
     were walked past (checked against the video). Any rise in severity
     (to approaching or critical) now has to hold for 3 frames
     (`ESCALATION_CONFIRM_S`, `speech_policy.py`) before it is spoken or
     shown; escalations then cut through the cooldown. Holding only
     critical wasn't enough -- the blips came back as "approaching". Costs
     ~0.13s on a true warning.
- **"0 false critical alerts" didn't survive testing on data it wasn't
  tuned against (2026-11-04).** The above was all measured on one session
  (`-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG`); running the fixed pipeline on the
  other two cached real SANPO sessions (`11tnj2tdX...`, heavy vehicular
  traffic; `2dVxdA...`, moderate traffic/high obstacle density) surfaced
  two more real false alarms, each a genuinely different mechanism:
  1. **Unlisted-concept noise (fixed).** Spark proposed "suitcase" for the
     `2dVxdA` walk. It matched neither word list in `motion_classes.py`,
     so it kept the pre-fix unconstrained behavior: velocity swung from
     +8 to -17.9 m/s between consecutive frames, with no suitcase visible
     in the corresponding camera frame -- the identical "trucks at 112
     m/s" failure, entering through a concept the hardcoded vocabulary
     never anticipated (and can't exhaustively, since Spark's vocabulary
     is open-ended). Fixed by asking Spark itself: any concept not covered
     by the hardcoded lists is classified once (static/movable + a max
     speed) via `SparkProposer.classify_motion()`
     (`generate_dashboard_data.py`'s `classify_unlisted_concepts`), one
     call per newly-seen unlisted concept, not per frame. Real call
     returned `suitcase -> movable, 5 m/s`. If the call itself fails, the
     concept defaults to movable/`DEFAULT_CAUTIOUS_MAX_SPEED_MPS` (3 m/s)
     rather than staying unprotected. The hardcoded lists always take
     precedence over a registered classification, so this can only add
     caution to a concept, never remove it from an already-verified one.
     Confirmed on the real walk: critical frames 23 -> 19, "suitcase"
     fully gone from the set of concepts ever flagged, every other
     concept's counts unchanged (the fix is surgical, not a behavior
     shift). The raw lifted position for this instance still jumps
     wildly -- the perception-level instability isn't fixed, only its
     ability to read as plausible velocity is.
  2. **Near-field depth drift (open, not fixed).** On the `11tnj2tdX`
     walk, a parked truck (close-range, large in frame, same tracked id
     from frame 0, never re-associated) produced a sustained ~1.2s
     critical episode: range closing 2.7m -> 1.8m while the wearer's own
     measured displacement over that second was only 0.5m and the wearer
     was stationary (`|v|=0.00`) for the back half of it. The object's own
     lifted position drifted ~0.67m toward the wearer in that second --
     slow and smooth enough to clear both the per-frame outlier-jump gate
     (tuned for sudden snaps, not a gradual crawl) and the least-squares
     velocity fit (which can't distinguish real slow motion from
     correlated noise of the same magnitude). `truck` already has a
     hardcoded 20 m/s ceiling, so this isn't a vocabulary gap -- it's
     larger absolute position error for large/close objects, which no
     ceiling this generous catches. No fix applied: a tighter floor risks
     suppressing a real, genuinely slow-moving vehicle, and that tradeoff
     needs real testing, not a last-minute patch. Documented as an open
     limitation.
  Net effect: the honest real-data count is not "0 false critical alerts"
  as a general claim -- it's 0 on the walk the fixes were tuned against,
  1 fixed and confirmed on a second walk, and 1 known, undocumented-no-
  longer, unfixed mechanism on a third.
- **Wording (2026-11-03).** Anything on a collision course (critical or
  approaching) is spoken in the short `describe()` form with its seconds
  (~1.6s of speech); the narrated sentence (2.2-2.6s) is kept for routine
  mentions, because speech duration comes straight out of lead time. The
  narrator is told the object's measured speed, or nothing if it isn't
  measured yet (it had called a van driving past at 2.4 m/s "stopped"),
  and is instructed never to imply safety (it had added "we're all good"
  and "we're clear" -- the spoken version of "PATH CLEAR").
  HUD: "NO ALERTS" instead of "PATH CLEAR" (the system can't see curbs,
  holes or untracked objects, so it can't claim a path is clear), and the
  object + side instead of "STOP" (describe, don't instruct).

Still not done:
- **`src/locus/app/pipeline.py`** -- the productionized async version of
  what `scripts/run_real_sequence.py` already proves works end-to-end as
  a one-shot script.
- **`src/locus/data/sanpo_loader.py` -- now real, not a stub.** SANPO
  directly fixes AEA's biggest real-data gap: dense depth. AEA's
  semi-dense point-cloud projection covers ~18% of a frame; SANPO's
  CREStereo depth map covers ~95-98% (measured on the session this loader
  defaults to, not assumed from docs). It's also an easier integration
  than expected: the whole 6TB dataset is a public, ungated GCS bucket
  (`gs://gresearch/sanpo_dataset/v0`, CC-BY 4.0) reachable over plain
  HTTPS -- no `gcloud` CLI, no GCP auth, no access-request gate at all
  (easier than AEA/SAM 3.1's HF gating). Per-session `description.json`
  ships already-pinhole intrinsics (zero distortion -- no fisheye
  correction needed, unlike AEA's Aria camera) and rich scene tags
  (`ego_motion`, traffic/obstacle/weather levels) that let a good walking
  session be picked by metadata instead of AEA's trial-and-error frame
  scanning. `config/default.yaml`'s `sanpo_sequences` defaults to one
  verified-good session (ego_motion WALKING, moderate traffic, medium
  obstacles, 539 frames). Two format facts had to be checked against real
  downloaded files rather than trusted from the dataset's own docs (which
  say npz): depth files are actually a 4-byte-prefixed raw float16 array,
  gzipped, not npz; and depth ships at a *different* resolution (1280x720)
  than its paired RGB frame (2208x1242) and needs resizing up before use.
  Covered by `tests/test_sanpo_loader.py`, including a test that confirms
  the real payoff (dense depth clears the mask-lifting pixel threshold on
  >90% of a grid of boxes across the frame, vs. AEA's frequent misses).
  **Update: the chest camera has now been used for exactly that.**
  Cross-validated `locus.geometry.frames`' velocity-derived body frame
  (the core design assumption behind the whole clock-face/body-vs-ego
  split) against SANPO's literal chest-camera-measured orientation for
  the first time. This surfaced and fixed two real bugs, documented in
  full in `sanpo_loader.py`'s module docstring (the short version, in
  order of discovery):
  1. Both rig cameras report the same image resolution but different
     focal lengths; `load_intrinsics` originally grabbed
     `session_camera_details[0]` unconditionally, which happened to be
     right for one camera and silently wrong for the other. Fixed by
     selecting on `model` name, cross-checked against each camera's own
     documented stereo baseline (chest ~120mm, head ~63mm) so a wrong
     mapping raises instead of silently producing right-shaped,
     wrong-valued intrinsics.
  2. **Corrected 2026-11-03 -- the original fix here was wrong.** It
     concluded SANPO's quaternion used local `-X` as forward and applied a
     correction matrix (`_R_SANPO_LOCAL_FROM_OPENCV`). That matrix had
     determinant -1 -- a reflection -- and the ~22 degree "median" that
     supported it was measured as `atan2(x, z)`, i.e. assuming Y-up in a
     world where the wearer walks along +Y, so it discarded the dominant
     motion component. The settled answer comes from multi-view depth
     reprojection (lift one frame's depth into world, reproject into a
     frame ~0.5s later, compare to that frame's own depth) over all 48
     signed axis permutations: **the raw quaternion is already
     OpenCV-camera (+X right, +Y down, +Z forward) in a right-handed Z-up
     world** -- identity scores 16% median relative depth error, the old
     matrix 45%, chance ~44%. The loader now applies no correction, and
     `SanpoSequence.world_down = (0, 0, -1)` (AEA is also Z-up; the
     synthetic world is OpenCV-axis, +Y down). `world_down` is threaded
     through `locus.geometry.frames` / `SpatialMemoryGraph`. This also
     resolves the up/right question previously left open here: Z is up,
     as the trajectory data said; the `..._y_up` feature name is the
     misleading part.

     User-visible consequence of the old matrix: on the shipped SANPO
     dashboard run, the reported side of an object matched the side of the
     image it was on only 55% of the time -- a coin flip. Replayed through
     the corrected pipeline: 351/351 (100%). Covered by
     `test_identity_rotation_convention_is_multiview_consistent` and
     `test_reported_side_matches_image_side`.

  Measured correctly (yaw in the plane perpendicular to real gravity), the
  head camera's velocity-derived heading agrees with the chest camera's
  measured heading at a **median of 14.7 degrees** on walking frames
  (larger excursions during turns, where a velocity-smoothed estimate
  legitimately lags real-time orientation). Covered by
  `tests/test_sanpo_loader.py::test_velocity_derived_heading_matches_chest_camera_measured_heading`.
  The wearer heading itself now comes from `WearerMotion`
  (`geometry/kinematics.py`): a 1s causal window of horizontal velocity,
  held through stops (271 of the session's 539 frames have the wearer
  standing still, where the old per-frame velocity collapsed the body
  frame), falling back to the camera's own forward before the first step.
- `requirements.lock.txt` / `docs/environment.md` -- not yet written.

## External resources (outside this repo, on the shared DGX box)
- Dataset: `/home/iairo-dgx1/vision_datasets/hackathon/aea/` -- 8 sequences
  downloaded and SHA1-verified (manifest: `aea/selected_sequences.json`).
  Assets kept per sequence: `video_main_rgb` (fisheye preview mp4, 20fps),
  `mps_slam_trajectories` (open/closed-loop pose), `mps_slam_calibration`
  (online per-camera calibration), `mps_slam_points` (semi-dense point
  cloud -- not yet consumed), `mps_slam_summary`, `mps_eye_gaze`,
  `annotations`. Deliberately not downloaded: `main_vrs` and
  `mps_artifacts` (large, not needed since RGB/pose come from the lighter
  preview+MPS outputs).
- SAM 3 / 3.1 code: `/home/iairo-dgx1/vision_datasets/hackathon/repos/sam3/`
  (public clone of `facebookresearch/sam3`; required to use the SAM 3.1
  checkpoint, which ships weights only, no code).

## Known open items
1. **`nvidia-cuda-nvrtc-cu12` keeps getting silently downgraded.** Any
   `pip install` that touches torch's dependency tree (installing
   `transformers`/`accelerate` for TTS did this at least once) resolves it
   back to the version torch declares (`12.8.93`), which cannot
   JIT-compile reduction kernels for GB10's compute capability 12.1 (see
   the SAM 3.1 section above for the actual nvrtc error). The fix is one
   line and doesn't touch anything else: `pip install
   "nvidia-cuda-nvrtc-cu12==12.9.86"`. **Check this first** if any real
   model inference suddenly fails with `nvrtc: error: invalid value for
   --gpu-architecture` after installing something new.
2. `use_fa3=False` (see above) means SAM 3.1 inference isn't running at
   its intended speed. Worth revisiting if live-demo FPS matters more than
   it currently does for a one-shot/offline proof of the full thread.
3. ~~Automatic concept selection isn't wired to a real model yet~~ --
   **done.** `SparkProposer` (`src/locus/perception/spark_proposer.py`) calls
   the real Meta Model API (muse-spark-1.3, OpenAI-compatible endpoint at
   `api.meta.ai/v1`) and is the default concept source in
   `scripts/generate_dashboard_data.py` (manual `--concepts` is now the
   override, not the default). It also re-asks at a few later checkpoints
   through a clip, not just frame 0 -- see the EdgeTAM/dashboard entry
   below for why a one-shot opening-frame proposal isn't enough on its own.
4. **PROJECT_PLAN.md's "run Meta open weights models" line listing Muse
   Glimmer 30B as a local model is stale.** Glimmer was the original plan
   for always-on narration, but its API column at
   https://dev.meta.ai/docs/models reads "None (self-host)" and it's
   absent from the hosted model list -- it isn't actually used anywhere
   in this repo. Muse Spark (`muse-spark-1.3`, hosted, already used for
   concept proposal) fills that narration role instead (`actuation/
   narrator.py`). Fix the submission writeup's model list before judging.

## Conventions
- Report ablation results (A/B/C/D, especially C vs D) honestly even if a
  sub-claim comes back negative -- see PROJECT_PLAN.md section 4. An
  honest negative sub-result reads as rigor, not failure.
- Don't guess geometry/calibration facts silently -- e.g. the AEA
  fisheye-undistortion pipeline in `aea_loader.py` required a few
  assumptions (native 2880x2880 sensor resolution, FISHEYE624 model,
  Gen1 device version) that were each cross-checked empirically (wrong
  guesses either errored immediately or produced a visibly-wrong
  undistorted image) before being trusted, rather than taken on faith.
