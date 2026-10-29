"""SAM 3.1 Object Multiplex session manager.

Native SAM 3.1 has no multi-concept-in-one-pass path: each concept slot
(~4-6 live at a time, chosen by concept_selector.Spark) gets its own
native video session, each multiplexing up to ~128 instances of that
one concept. This module owns those parallel sessions and their
per-concept track IDs.

See docs/PROJECT_PLAN.md section 5/6/8 for the architecture and the
day-1 FPS benchmark this depends on.
"""

from dataclasses import dataclass


@dataclass
class ConceptSession:
    concept: str
    session_id: str | None = None


class SamSessionManager:
    """Owns one native SAM 3.1 Object Multiplex session per live concept slot."""

    def __init__(self, concept_slots: list[str]):
        self.concept_slots = concept_slots
        self.sessions: dict[str, ConceptSession] = {}

    def start_sessions(self, video_path: str):
        """Start one SAM 3.1 session per concept slot against the given video/frame source."""
        raise NotImplementedError

    def propagate(self, frame_index: int):
        """Advance all concept sessions to the given frame, returning masks + track IDs per concept."""
        raise NotImplementedError
