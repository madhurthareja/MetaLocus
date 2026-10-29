"""A ConceptProposer that already knows which concepts exist in a synthetic
scenario, instead of looking at frame.rgb. Stands in for SparkProposer in
the eval harness so conditions A-D can run without a VLM call per frame.
"""

from __future__ import annotations

from locus.data.synthetic import GroundTruth
from locus.types import Frame


class MockConceptProposer:
    def __init__(self, ground_truth: GroundTruth, max_concepts: int = 6):
        self._concepts = sorted({obj.concept for obj in ground_truth.objects})
        self._max_concepts = max_concepts

    def propose(self, frame: Frame, graph_summary: dict) -> list[str]:
        return self._concepts[: self._max_concepts]
