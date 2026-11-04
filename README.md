# Locus

Meta Global AI Developer Hackathon (Indie track). A hazard-warning system
for outdoor blind/low-vision navigation: first-person video → SAM 3.1
Object Multiplex segmentation → a persistent, geometric 3D scene graph →
clock-face audio cues (stereo pan + TTC pulse) and ≤6-word spoken
directives. Primary dataset is Google's SANPO — built specifically for
this problem and already powering Project Guideline in production, so the
pitch and the benchmark are the same problem statement, not a proxy (see
`docs/PROJECT_PLAN.md` §1-3). AEA (indoor) runs the identical pipeline
unchanged as a cross-dataset generalization check, not the headline.

**Thesis:** frame-level VLM captioning fails systematically on spatial
questions once the referent leaves view; grounding detections in a
persistent, segmentation-derived world-frame graph recovers that
capability. The A/B/C/D ablation in `src/locus/eval/` is built to prove
(or honestly disprove) this, including the controlled C-vs-D comparison
(box-lifted vs. mask-lifted instances through the identical persistent
graph) that isolates whether masks specifically — not just persistence —
matter.

Full strategic rationale (pitch, benchmark design, model stack, hackathon
rules, build schedule): **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**.
Day-to-day project context for Claude Code: **[docs/CLAUDE.md](docs/CLAUDE.md)**.

## Status

| Piece | State |
|---|---|
| Geometry (3 reference frames, lifting, kinematics) | Done, tested |
| Memory graph (persistence, decay, eviction, re-association) | Done, tested |
| Synthetic data generator | Done — unit-test harness only, not a benchmark substitute |
| Eval harness (mocks + A/B/C/D conditions + metrics) | Done, tested end-to-end on synthetic scenarios |
| Audio actuation (stereo-pan + TTC-pulse cues, spoken directives) | Done, tested |
| Real SANPO loader (dense depth, pinhole intrinsics, no gate) — **primary dataset** | **Done** — ~95-98% depth coverage, real outdoor hazards (e.g. a car at 1.8s TTC); see `docs/CLAUDE.md` |
| Real AEA loader (pose + fisheye-undistorted RGB) — secondary, cross-dataset check | Done, verified on all 8 downloaded sequences |
| Real AEA sparse depth (semi-dense point cloud → per-frame z-buffer) | Done, ~18% pixel coverage (vs. SANPO's ~96%) — a known limitation of AEA specifically |
| SAM 3.1 native segmenter | **Done** — real session wiring, run successfully on both real SANPO and AEA frames; mask quality verified visually, not just "it ran" |
| EdgeTAM live segmenter (`--segmenter edgetam`) | **Done** — genuinely causal/streaming alternative to SAM 3.1's batch pass, for the real-time path; ~10.5 FPS on real SANPO frames (vs. SAM 3.1's 0.17-0.68 FPS). Lower accuracy on long clips (no occlusion re-identification), substantially mitigated by periodic re-grounding — see `docs/CLAUDE.md` for real measured before/after numbers |
| Spoken-directive pacing + dashboard playback speed | **Done** — at most one directive spoken per cooldown window (safety escalation always cuts in); cut one real session from 252 unique spoken clips to 23. See `docs/CLAUDE.md` |
| SANPO left/right + hazard-alert noise | **Found and fixed** — SANPO's pose convention was being mirrored (reported side matched image side 55% of the time → 100%); ~94% of critical alerts were velocity noise (static-object jitter, segmenter identity swaps, walking past things) → class-dependent memory, per-segment mover velocity, closest-approach gate. Critical frames on the real session 342 → 0 (the last two were one-frame blips on parked vehicles, removed by a 3-frame hold). See `docs/CLAUDE.md` |
| Speech novelty rule | **Done** — each object announced once, again only on a real change (severity up, or a 2+ clock-hour move ahead); spoken cues on the real session 25 → 12 |
| Complete real-data thread (pose + RGB + depth + segmentation + memory graph + TTC + directives + spoken TTS) | `scripts/generate_dashboard_data.py --dataset {sanpo,aea} [--segmenter sam3.1\|edgetam]` is now the primary entry point (produces a full dashboard session); `scripts/run_real_sequence.py` was the original one-shot proof and still works but is superseded by it |
| TTS (spoken directives) | **Done** — Meta SeamlessM4T v2 (`src/locus/actuation/tts.py`); Voicebox/Audiobox were investigated and are not actually available (see `docs/CLAUDE.md`) |
| Duplicate-node bug (SAM 3.1 double-tracking one object → two memory-graph nodes) | **Found and fixed** on real data — see `docs/CLAUDE.md` |
| `app/pipeline.py` (the productionized async version of the above) | Not yet built |

Run the test suite: `pytest tests/ -v` (55 tests, all passing as of this
writing — spans geometry, memory, the eval harness's own thesis-regression
test, audio actuation, TTS wiring, and real-data integration tests for
both SANPO and AEA that skip cleanly if the data/network isn't reachable
from a given machine).

Run the ablation harness on synthetic scenarios: `python -m locus.eval.harness`.

Run the complete real thread end to end (real pose, real RGB, real depth,
real SAM 3.1 segmentation, persistent memory graph, TTC, spoken directives)
on either dataset:
```
# SANPO (outdoor, primary) -- real street hazards
python scripts/run_real_sequence.py --dataset sanpo --n-frames 40 --concepts "car,person,truck"

# AEA (indoor, cross-dataset check) -- identical pipeline, different sensor
python scripts/run_real_sequence.py --dataset aea loc5_script4_seq6_rec1 --n-frames 30 --concepts "chair,couch,table"
```

## Where things live

This repo (`/home/madhur/Desktop/hackathon`) holds the project's own code,
config, and docs — kept here deliberately rather than a fresh directory,
since this repo already has git history and a pushed GitHub remote.

External resources live elsewhere on this shared DGX box, outside the repo:

| What | Where |
|---|---|
| SANPO cache (downloaded per-frame on demand from Google's public GCS bucket) | `/home/iairo-dgx1/vision_datasets/hackathon/sanpo_cache/` |
| AEA dataset (8 selected sequences: RGB preview, SLAM trajectories/points/calibration, eye gaze, annotations) | `/home/iairo-dgx1/vision_datasets/hackathon/aea/` (manifest: `aea/selected_sequences.json`) |
| SAM 3 / SAM 3.1 code (`facebookresearch/sam3`, public clone) | `/home/iairo-dgx1/vision_datasets/hackathon/repos/sam3/` |
| SAM 3.1 Object Multiplex checkpoint (`facebook/sam3.1`, gated on Hugging Face) | gated access approved; pulled via `hf` |

Paths are also recorded in [`config/default.yaml`](config/default.yaml).

## Repo structure

```
src/locus/
  types.py          Detection, Pose, Frame, Instance3D, Node, Edge, Query, Answer
  config.py         config/default.yaml + LOCUS_<SECTION>__<KEY> env overrides
  geometry/          3 reference frames (ego/body/world), intrinsics, mask/box lifting, kinematics/TTC
  memory/            persistent spatial graph: decay, eviction, spatial re-association, edges
  perception/        Segmenter + ConceptProposer protocols; mocks (ground-truth-driven); SAM 3.1 native (batch) + EdgeTAM (live/streaming) backends; real Spark proposer
  data/              synthetic scenario generator (unit-test harness); real AEA loader (sparse depth); real SANPO loader (dense depth)
  eval/              A/B/C/D ablation conditions, metrics, question generation, CLI harness
  actuation/         stereo-pan + TTC-pulse audio cues; ≤6-word spoken directives
  app/               (not yet built) pipeline tying a real sequence to perception → memory → actuation
tests/              one file per module area; the eval test is a thesis-regression check, not just a unit test
scripts/            env setup, SAM 3.1 smoke test (not yet filled in)
config/             default.yaml (paths, sequence list, thresholds)
docs/               PROJECT_PLAN.md, CLAUDE.md
```

## Setup

```
bash scripts/setup_env.sh      # conda env 'hackathon', Python 3.12, torch cu128, SAM3, projectaria_tools
hf auth login                  # once facebook/sam3.1 gated access is approved (already done)
pip install -e ".[dev]"
pytest tests/ -v
```
