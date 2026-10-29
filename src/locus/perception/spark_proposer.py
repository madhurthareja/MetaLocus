"""Muse Spark 1.3 concept proposer, via the Meta Model API (~0.5 Hz)."""

from __future__ import annotations

from locus.types import Frame


class SparkProposer:
    def __init__(self, max_concepts: int = 6):
        self.max_concepts = max_concepts

    def propose(self, frame: Frame, graph_summary: dict) -> list[str]:
        raise NotImplementedError
