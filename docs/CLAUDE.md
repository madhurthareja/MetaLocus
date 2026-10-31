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
Perception Encoder) is still the untested unknown -- `scripts/smoke_test_sam31.py`
needs to actually be filled in and run.

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
  2. SANPO's quaternion uses a different local "forward" axis than the
     OpenCV convention (`+Z` forward) that `Intrinsics`/the rest of
     `locus.geometry` assumes everywhere else, including for AEA. This is
     invisible to any RGB/depth visual check -- it only shows up when you
     compare a velocity-derived heading to the camera's own
     orientation-derived heading. Testing all six signed local axes
     against the head camera's own smoothed velocity found local `-X`
     gives a clean, small residual (median ~22 degrees); every other axis
     gave ~100+ degrees of essentially uncorrelated error. The up/right
     axes remain a genuinely open question (not just unverified algebra):
     the dataset's own `..._right_handed_y_up` naming implies translation
     Y is vertical, but this loader's own `fixed_camera_poses.csv` data
     says otherwise (Z is the axis that stays bounded across a session
     tagged elevation-flat; Y is the one that grows by 14m). Diffing
     `fixed_camera_poses.csv` against the dataset's own *unfixed*
     `camera_poses.csv` ruled out a simple axis permutation as the
     explanation -- the two trajectories diverge progressively, meaning
     "fixed" is a genuinely re-estimated trajectory, not a relabeling.
     This loader currently assumes up=Z (matching its own data) over
     up=Y (the literal feature name), but this specifically affects only
     lateral/vertical placement of lifted 3D points, not depth-along-ray
     or heading -- which is why the validation below still means what it
     says despite this being unresolved.

  With the forward-axis fix applied, the validation succeeded on the part
  it can check: the head camera's velocity-derived heading agrees with
  the chest camera's real measured heading at a median of ~22 degrees
  (larger excursions during turns, where a velocity-smoothed estimate
  legitimately lags real-time orientation). Covered by
  `tests/test_sanpo_loader.py::test_velocity_derived_heading_matches_chest_camera_measured_heading`.
  This is the first time this project has validated the body-frame
  approximation against anything other than "it seems reasonable" --
  worth being honest that it's a median-level agreement with real
  scatter, not an exact match, and that the up/right axis question would
  need the dataset maintainers (sanpo_dataset@google.com) or a
  lateral-position-specific test to fully resolve.
- `scripts/smoke_test_sam31.py` -- still raises `NotImplementedError`.
  This is the one open question that's actually load-bearing for the live
  demo: real per-session FPS with 4-6 concurrent SAM 3.1 concept sessions
  on GB10, needed before locking in live concept count / frame rate.
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
3. Automatic concept selection (the Spark/SparkProposer side of the
   architecture) isn't wired to a real model yet -- `scripts/run_real_sequence.py`
   takes a manually-specified concept list. Needed once live/automatic
   concept proposing (vs. a fixed list) matters.

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
