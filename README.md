# Locus

Locus is a hazard-warning system for outdoor navigation by blind and
low-vision users. It takes first-person video and turns it into spoken,
clock-face directions ("car at 2 o'clock, 3s") about things that matter.

The pipeline: first-person video → SAM 3.1 Object Multiplex segmentation
→ a persistent, geometric 3D scene graph → time-to-collision and
closest-point-of-approach → stereo-panned audio cues and short spoken
directives. The primary dataset is Google's SANPO, built specifically for
outdoor blind/low-vision navigation and already powering Project
Guideline in production. AEA (indoor) runs the identical pipeline as a
secondary cross-dataset check.

Built for the Meta Global AI Developer Hackathon (Indie track).

## Why persistence

A camera-only system forgets an object the instant it leaves the frame.
Locus keeps a world-anchored memory graph instead: an object stays
tracked — with decaying confidence, eventually evicted — after it's out
of view, so turning your head doesn't lose track of what's beside you.
What's safe to remember differs by object: a pole's position is good
indefinitely, a car's is only good for a second or two, so static and
moving concepts are tracked with different rules.

The `src/locus/eval/` ablation ladder (conditions A–D) is the evidence
for this: a stateless baseline, a short sliding window, and the full
persistent graph with box- vs. mask-lifted instances, run against the
same synthetic ground truth.

## Repo structure

```
src/locus/
  types.py          Detection, Pose, Frame, Instance3D, Node, Edge, Query, Answer
  config.py         config/default.yaml + LOCUS_<SECTION>__<KEY> env overrides
  geometry/          3 reference frames (ego/body/world), intrinsics, mask/box lifting, kinematics/TTC
  memory/            persistent spatial graph: decay, eviction, spatial re-association, edges
  perception/        Segmenter + ConceptProposer protocols; mocks (ground-truth-driven); SAM 3.1 native (batch) + EdgeTAM (live/streaming) backends; real Spark proposer
  data/              synthetic scenario generator; real AEA loader (sparse depth); real SANPO loader (dense depth)
  eval/              A/B/C/D ablation conditions, metrics, question generation, CLI harness
  actuation/         stereo-pan + TTC-pulse audio cues; ≤6-word spoken directives; speech pacing policy
  app/               (not yet built) pipeline tying a real sequence to perception → memory → actuation
tests/              one file per module area
scripts/            env setup, dashboard data generation, smoke tests
dashboard/          a browser UI that replays a generated session: camera + segmentation, the 3D memory graph, hazard timeline, system diagnostics
config/             default.yaml (paths, sequence list, thresholds)
```

## Setup

```
bash scripts/setup_env.sh      # conda env 'hackathon', Python 3.12, torch cu128, SAM3, projectaria_tools
hf auth login                  # once facebook/sam3.1 gated access is approved
pip install -e ".[dev]"
pytest tests/ -v
```

## Usage

Run the synthetic A/B/C/D ablation:
```
python -m locus.eval.harness
```

Run the complete real thread end to end (real pose, real RGB, real depth,
real segmentation, persistent memory graph, TTC, spoken directives) on
either dataset:
```
# SANPO (outdoor, primary) -- real street hazards
python scripts/run_real_sequence.py --dataset sanpo --n-frames 40 --concepts "car,person,truck"

# AEA (indoor, cross-dataset check) -- identical pipeline, different sensor
python scripts/run_real_sequence.py --dataset aea loc5_script4_seq6_rec1 --n-frames 30 --concepts "chair,couch,table"
```

Generate a full dashboard session and view it in the browser:
```
python scripts/generate_dashboard_data.py --dataset sanpo --segmenter edgetam
python dashboard/app.py   # then open http://localhost:5050
```

## Where things live

This repo (`/home/madhur/Desktop/hackathon`) holds the project's own code
and config. External resources live elsewhere on this shared DGX box,
outside the repo:

| What | Where |
|---|---|
| SANPO cache (downloaded per-frame on demand from Google's public GCS bucket) | `/home/iairo-dgx1/vision_datasets/hackathon/sanpo_cache/` |
| AEA dataset (8 selected sequences: RGB preview, SLAM trajectories/points/calibration, eye gaze, annotations) | `/home/iairo-dgx1/vision_datasets/hackathon/aea/` (manifest: `aea/selected_sequences.json`) |
| SAM 3 / SAM 3.1 code (`facebookresearch/sam3`, public clone) | `/home/iairo-dgx1/vision_datasets/hackathon/repos/sam3/` |
| SAM 3.1 Object Multiplex checkpoint (`facebook/sam3.1`, gated on Hugging Face) | gated access approved; pulled via `hf` |

Paths are also recorded in [`config/default.yaml`](config/default.yaml).
