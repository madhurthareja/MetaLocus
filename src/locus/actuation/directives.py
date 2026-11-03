"""Edge -> short spoken directive. At most six words: this is heard while
walking, not read, so it has to resolve in about a second of TTS.

This is deliberately the fast, local, no-API-call path -- the one always
used for a genuinely imminent hazard (ttc_s < HAZARD_TTC_THRESHOLD_S),
where a few seconds of margin don't exist to spend on a network round
trip, and the only fallback if the warmer, narrated path (locus.actuation.
narrator.SparkNarrator, used for everything with real margin) is
unavailable or fails."""

from __future__ import annotations

import numpy as np

from locus.types import Edge

HAZARD_TTC_THRESHOLD_S = 2.0


def clock_hour(bearing_rad: float) -> int:
    degrees = np.degrees(bearing_rad) % 360.0
    clock = round(degrees / 30.0) % 12
    return clock if clock != 0 else 12


def is_critical(edge: Edge) -> bool:
    return edge.ttc_s is not None and edge.ttc_s < HAZARD_TTC_THRESHOLD_S


def describe(concept: str, edge: Edge) -> str:
    clock = clock_hour(edge.bearing_rad)
    if is_critical(edge):
        return f"{concept}, {clock} o'clock, {edge.ttc_s:.0f}s"
    return f"{concept} at {clock} o'clock"
