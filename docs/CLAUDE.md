# Project context for Claude Code

Autonomous Egocentric Hazard Agent -- Meta Global AI Developer Hackathon,
Indie track, team of 2. Full plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Note:** this file lives in `docs/` rather than the repo root, so it is
not auto-loaded as Claude Code project memory. Move or symlink it to the
repo root if auto-load is wanted.

## Timeline
Kickoff Oct 26, submission due **Nov 4, 9:00 PM ET**. 2 of 9 hackathon days
were already gone before building started; see PROJECT_PLAN.md section 10
for the day-by-day schedule mapped to actual calendar dates.

## Hardware
NVIDIA DGX Spark (GB10, aarch64/Grace-Blackwell), 128GB unified memory,
CUDA 13.0 toolkit installed. Memory capacity is not the constraint;
per-concept SAM 3.1 session throughput (each reruns the Perception
Encoder) is the untested unknown -- day-1 priority is
`scripts/smoke_test_sam31.py`.

## Key architecture decisions (see PROJECT_PLAN.md for full rationale)
- **SAM 3.1 Object Multiplex, required by the hackathon's Meta Model API
  tooling.** Native path is one session per concept (no multi-concept
  single pass), 4-6 live concept slots chosen by Spark.
- **AEA (Project Aria Everyday Activities), not EgoObjects.** AEA gives
  free depth (semi-dense point cloud), 6-DoF pose, and calibration --
  needed for real hazard distances. EgoObjects has neither.
- **No GATv2 / learned GNN.** Time-to-collision is `distance / closing
  speed`, computed geometrically. No training labels exist in the
  timeline; an undertrained GNN is worse than honest geometry.
- **Open-loop trajectory, not closed-loop.** Mirrors what an on-device
  system would actually have at inference time.
- Glimmer runs **always-on** (hazard_loop), independent of the
  query-answering layer (query.py / Wonder).

## External resources (outside this repo, on the shared DGX box)
- Dataset: `/home/iairo-dgx1/vision_datasets/hackathon/aea/` -- 8 sequences
  downloaded and SHA1-verified (manifest: `aea/selected_sequences.json`).
  Assets kept per sequence: `video_main_rgb`, `mps_slam_trajectories`,
  `mps_slam_calibration`, `mps_slam_points`, `mps_slam_summary`,
  `mps_eye_gaze`, `annotations`. Deliberately **not** downloaded:
  `main_vrs` (multi-GB full recording) and `mps_artifacts` (large bulk
  output) -- not needed since RGB comes from the lightweight preview mp4
  and depth/pose come from the MPS outputs.
- SAM 3 / 3.1 code: `/home/iairo-dgx1/vad_data/hackathon/repos/sam3/`
  (public clone of `facebookresearch/sam3`; this repo's code is required
  to use the SAM 3.1 checkpoint, which ships weights only, no code).

## Outstanding blockers as of this writing
1. `facebook/sam3.1` on Hugging Face is gated (manual review) -- access
   requested, not yet confirmed approved.
2. The cached HF token on this box was invalid/expired; needs
   `hf auth login` (or `HF_TOKEN` env var) once access is approved.
3. Python 3.12 conda env (`scripts/setup_env.sh`) not yet built --
   conda ToS for the default channels has now been accepted, this is
   unblocked.
4. TTS backend for the hazard-alert loop is undecided (Muse's lineup only
   has Voice *Transcribe*, which is STT, not TTS).

## Conventions
- All `src/hazard_agent/` modules are currently scaffolding: signatures
  and docstrings only, `raise NotImplementedError`. Fill in against
  PROJECT_PLAN.md's architecture (section 6), not ad hoc.
- Report ablation results (A/B/C/D, especially C vs D) honestly even if a
  sub-claim comes back negative -- see PROJECT_PLAN.md section 4.
