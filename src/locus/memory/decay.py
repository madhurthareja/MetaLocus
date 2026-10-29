"""Exponential confidence decay by half-life.

Recomputed fresh from elapsed time since last observation each call,
rather than compounded across repeated step() calls -- avoids
accumulating floating-point drift for nodes that go unseen a long time.
"""

from __future__ import annotations


def decayed_confidence(elapsed_s: float, half_life_s: float) -> float:
    if elapsed_s <= 0:
        return 1.0
    return 0.5 ** (elapsed_s / half_life_s)
