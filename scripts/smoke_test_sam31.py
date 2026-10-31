"""Day-1 smoke test: confirm SAM 3.1 Object Multiplex actually runs on the
DGX Spark (GB10, aarch64), and clock real wall-clock throughput across
concept-session counts before locking the live-demo frame rate or concept
count into the architecture. See docs/PROJECT_PLAN.md sections 8 and 9.

Run from the 'hackathon' conda env after scripts/setup_env.sh and after
`hf auth login` has access to the gated facebook/sam3.1 checkpoint.

This was a stub (`raise NotImplementedError`) until the real pipeline was
working end-to-end. Filled in now that it is: src/locus/perception/
sam3_native.py's own docstring already states the real constraint --
prepare() runs one full start_session/add_prompt/propagate_in_video/
close_session cycle PER CONCEPT, strictly sequentially. There is no
concurrent or multiplexed execution across concepts in this codebase's
integration, whatever "Object Multiplex" means at the model level. So the
real question this smoke test answers isn't "does throughput degrade under
concurrency" -- it's "how does wall-clock scale with concept count", which
is what's actually measured below: the same small real frame window run
through prepare() at 1, 3, and 6 concepts.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import torch

from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.data.sanpo_loader import load_sequence
from locus.perception.sam3_native import Sam31NativeSegmenter

N_FRAMES = 24
CONCEPT_COUNTS = [1, 3, 6]
# The real concepts Spark proposed for this project's primary SANPO session
# (dashboard/data/sanpo_.../timeline.json's own "concepts" field) -- using
# the real concept list rather than placeholder words so the benchmark
# reflects actual prompt difficulty, not just prompt count.
ALL_CONCEPTS = ["pole", "van", "tree", "fire hydrant", "trash bin", "truck"]


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

    results = []
    for n in CONCEPT_COUNTS:
        concepts = ALL_CONCEPTS[:n]
        segmenter = Sam31NativeSegmenter()
        segmenter.start(concepts)
        frame_dir = Path(tempfile.mkdtemp(prefix="smoke_sam31_"))
        try:
            t0 = time.time()
            segmenter.prepare(frames, frame_dir)
            elapsed = time.time() - t0
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
        fps = len(frames) / elapsed
        per_concept_s = elapsed / n
        print(
            f"  {n} concept(s) {concepts}: {elapsed:.1f}s total, "
            f"{fps:.2f} frames/s (aggregate across all {n} sessions), "
            f"{per_concept_s:.2f}s/concept-session"
        )
        results.append(
            {
                "n_concepts": n,
                "concepts": concepts,
                "n_frames": len(frames),
                "elapsed_s": round(elapsed, 2),
                "fps_aggregate": round(fps, 3),
                "seconds_per_concept_session": round(per_concept_s, 3),
            }
        )

    out = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "n_frames_per_run": len(frames),
        "dataset": "sanpo",
        "session_id": session_id,
        "results": results,
        "note": (
            "prepare() runs one full session per concept, strictly sequentially "
            "(no concurrent/multiplexed execution across concepts in this "
            "integration) -- elapsed scales with concept count because of that "
            "loop, not GPU contention. These are forward+reverse propagation "
            "wall-clock numbers on a small real frame window, not a production "
            "per-frame latency SLA."
        ),
    }
    out_path = Path("dashboard/data/sam31_smoke_test.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"done in {time.time() - start:.1f}s")
