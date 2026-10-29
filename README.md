# Autonomous Egocentric Hazard Agent

Meta Global AI Developer Hackathon (Indie track). Edge-compute assistive
navigation: first-person video -> SAM 3.1 Object Multiplex segmentation ->
persistent 3D scene graph -> an always-on local agent (Muse Glimmer) that
warns a visually impaired user of physical hazards via TTS.

Full project plan, architecture, decisions and rationale: **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**.

## Where things live

This repo (`/home/madhur/Desktop/hackathon`) holds the project's own code,
config, and docs. Two external resources live elsewhere on this shared DGX
box, outside the repo:

| What | Where |
|---|---|
| AEA dataset (8 selected sequences: RGB preview, SLAM trajectories/points/calibration, eye gaze, annotations) | `/home/iairo-dgx1/vision_datasets/hackathon/aea/` (manifest: `aea/selected_sequences.json`) |
| SAM 3 / SAM 3.1 code (`facebookresearch/sam3`, public clone) | `/home/iairo-dgx1/vad_data/hackathon/repos/sam3/` |
| SAM 3.1 Object Multiplex checkpoint (`facebook/sam3.1`, gated on Hugging Face) | pulled via `hf` once gated access is approved for this account |

Paths are also recorded in [`configs/default.yaml`](configs/default.yaml).

## Repo structure

```
src/hazard_agent/
  perception/      SAM 3.1 session manager (one native session per concept slot) + Spark concept selector
  geometry/         fisheye undistortion/intrinsics, mask -> 3D position via AEA's point cloud
  graph/            persistent scene graph (nodes, confidence decay, geometric edges) + time-to-collision
  agent/            always-on hazard loop (Glimmer -> TTS) and the Wonder query-answering layer
  data/             AEA sequence loader
  eval/             5-type question taxonomy + A/B/C/D ablation harness
scripts/            env setup, day-1 SAM 3.1 smoke test
configs/            default.yaml (paths, sequence list, thresholds)
docs/               PROJECT_PLAN.md, CLAUDE.md
```

Everything under `src/` is currently scaffolding (signatures + docstrings,
`raise NotImplementedError`) reflecting the architecture in the plan -- not
yet implemented.

## Setup

```
bash scripts/setup_env.sh      # conda env 'hackathon', Python 3.12, torch cu128, SAM3, projectaria_tools
hf auth login                  # once facebook/sam3.1 gated access is approved
python scripts/smoke_test_sam31.py   # day-1: confirm SAM 3.1 runs on GB10, measure FPS
```
