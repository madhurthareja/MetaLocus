"""Muse Spark 1.3 concept proposer (~0.5Hz).

Picks which 4-6 concepts currently get a live SAM 3.1 session, and
parses ad-hoc queries into (anchor, relation, target) triples for the
query layer (agent/query.py).
"""


class ConceptSelector:
    def __init__(self, max_concepts: int = 6):
        self.max_concepts = max_concepts

    def select_concepts(self, context) -> list[str]:
        """Return the concept slots that should have a live SAM 3.1 session right now."""
        raise NotImplementedError

    def parse_query(self, query: str) -> tuple[str, str, str]:
        """Parse a free-form query into (anchor, relation, target)."""
        raise NotImplementedError
