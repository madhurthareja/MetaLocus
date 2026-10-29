"""Geometric time-to-collision: distance / closing speed.

Deliberately not a learned GNN (a GATv2 was considered and dropped --
no training signal available in the hackathon timeline, and an
undertrained GNN is a worse hazard detector than honest geometry). This
is fully explainable and costs zero training time.
"""


def time_to_collision(distance: float, closing_speed: float) -> float | None:
    """Return seconds to collision, or None if not closing (closing_speed <= 0)."""
    if closing_speed <= 0:
        return None
    return distance / closing_speed
