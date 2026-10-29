"""Segmenter protocol. Every real backend and MockSegmenter implement this
exactly, so swapping one for the other is the only change needed anywhere
else in the pipeline.
"""

from __future__ import annotations

from typing import Protocol

from locus.types import Detection, Frame


class Segmenter(Protocol):
    def start(self, concepts: list[str]) -> None: ...

    def update_concepts(self, concepts: list[str]) -> None: ...

    def step(self, frame: Frame) -> list[Detection]: ...

    def close(self) -> None: ...
