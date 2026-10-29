"""SAM 3 (not 3.1) via HF Transformers Sam3Video -- true multi-concept
single pass, shared vision-encoder features across prompts.

Not the segmenter this hackathon uses (the hackathon's Meta Model API
tooling requires SAM 3.1 Object Multiplex -- see sam3_native.py), but
kept as a documented stub for the half-day stretch-goal micro-benchmark
comparing the two session strategies at this project's object count
(docs/PROJECT_PLAN.md section 5).
"""

from __future__ import annotations

from locus.types import Detection, Frame


class Sam3HfSegmenter:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def start(self, concepts: list[str]) -> None:
        raise NotImplementedError

    def update_concepts(self, concepts: list[str]) -> None:
        raise NotImplementedError

    def step(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
