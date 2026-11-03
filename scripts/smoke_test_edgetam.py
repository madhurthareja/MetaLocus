"""Smoke test for EdgeTamSegmenter (src/locus/perception/edgetam_segmenter.py)
against real SANPO frames on the DGX Spark (GB10) -- confirms the live,
causal, frame-by-frame segmentation path actually runs end to end (grounding
via one SAM 3.1 add_prompt call per concept, then genuinely-streaming
EdgeTAM tracking), and clocks real per-frame latency once tracking is
steady, separately from the one-time model-load and per-concept grounding
costs that only happen at the start of a session.

Run from the 'hackathon' conda env, same prerequisites as
scripts/smoke_test_sam31.py (gated facebook/sam3.1 checkpoint access, since
the default grounder is Sam31NativeSegmenter.ground_text).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.data.sanpo_loader import load_sequence
from locus.perception.edgetam_segmenter import EdgeTamSegmenter

N_FRAMES = 60
# Concepts already confirmed present in this session by the real SAM 3.1
# pipeline run (dashboard/data/sanpo_.../timeline.json), so grounding
# success reflects real scene content, not a guessed prompt.
CONCEPTS = ["pole", "van", "tree"]


def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    config = load_config(DEFAULT_CONFIG_PATH)
    session_id = config["sanpo_sequences"][0]
    print(f"loading {N_FRAMES} real frames from SANPO session {session_id!r}...")
    seq = load_sequence(session_id, config=config)
    seq.prefetch(max_frames=N_FRAMES)
    frames = list(seq.iter_frames(max_frames=N_FRAMES))
    print(f"loaded {len(frames)} real frames")

    segmenter = EdgeTamSegmenter()

    t_load0 = time.time()
    segmenter.start(CONCEPTS)
    load_elapsed = time.time() - t_load0
    print(f"model load + session init: {load_elapsed:.1f}s")

    per_frame_s = []
    grounded_at: dict[str, int] = {}
    detections_per_frame = []
    for i, frame in enumerate(frames):
        t0 = time.time()
        dets = segmenter.step(frame)
        dt = time.time() - t0
        per_frame_s.append(dt)
        detections_per_frame.append(len(dets))
        for d in dets:
            grounded_at.setdefault(d.concept, i)
        print(f"  frame {i}: {dt:.3f}s, {len(dets)} detection(s) -- {[d.concept for d in dets]}")

    segmenter.close()

    # Frames before every concept either grounded or exhausted its attempts
    # pay SAM 3.1 grounding cost on top of EdgeTAM's own forward pass;
    # steady-state is measured only once every concept has resolved
    # (grounded or never found), so it reflects EdgeTAM's actual per-frame
    # tracking cost, not SAM 3.1's.
    settle_idx = max(grounded_at.values(), default=0) + 1
    steady = per_frame_s[settle_idx:]
    out = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "dataset": "sanpo",
        "session_id": session_id,
        "n_frames": len(frames),
        "concepts": CONCEPTS,
        "model_load_s": round(load_elapsed, 2),
        "grounded_at_frame": grounded_at,
        "per_frame_s_including_grounding": [round(x, 4) for x in per_frame_s],
        "steady_state": {
            "n_frames": len(steady),
            "mean_s": round(sum(steady) / len(steady), 4) if steady else None,
            "fps": round(len(steady) / sum(steady), 2) if steady else None,
        },
        "detections_per_frame": detections_per_frame,
        "note": (
            "Real per-frame wall-clock on the DGX Spark (GB10), causal/streaming "
            "(no look-ahead) EdgeTAM tracking seeded by one SAM 3.1 add_prompt "
            "call per concept (no propagation). steady_state excludes frames "
            "before every concept either grounded or was given up on -- this "
            "is a benchmark result, not production telemetry."
        ),
    }
    out_path = Path("dashboard/data/edgetam_smoke_test.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"steady-state: {out['steady_state']}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"done in {time.time() - start:.1f}s")
