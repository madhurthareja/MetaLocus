"""Persistent allocentric scene graph.

Nodes are tracked objects (SAM track ID + 3D position), with confidence
that decays once a node exits the field of view. Edges (left-of,
right-of, behind, distance, closing-speed) are computed geometrically
from node positions -- never learned. This is the "graph remembers what
left the frame" capability neither SAM nor a frame-level VLM has.
"""

from dataclasses import dataclass, field


@dataclass
class SceneNode:
    track_id: str
    concept: str
    position: tuple[float, float, float]
    confidence: float = 1.0
    last_seen_frame: int = 0


class SceneGraph:
    def __init__(self, decay_rate: float = 0.0):
        self.nodes: dict[str, SceneNode] = {}
        self.decay_rate = decay_rate

    def update(self, frame_index: int, observations: list[SceneNode]):
        """Upsert observed nodes and decay confidence for nodes not seen this frame."""
        raise NotImplementedError

    def relation(self, anchor_id: str, target_id: str) -> str:
        """Compute a geometric spatial relation (left-of/right-of/behind/...) between two nodes."""
        raise NotImplementedError
