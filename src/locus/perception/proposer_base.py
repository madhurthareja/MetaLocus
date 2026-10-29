"""Concept proposer protocol."""

from __future__ import annotations

from typing import Protocol

from locus.types import Frame


class ConceptProposer(Protocol):
    def propose(self, frame: Frame, graph_summary: dict) -> list[str]: ...
