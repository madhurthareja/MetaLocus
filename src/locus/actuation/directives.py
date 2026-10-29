"""Edge -> short spoken directive. At most six words: this is heard while
walking, not read, so it has to resolve in about a second of TTS."""

from __future__ import annotations

import numpy as np

from locus.types import Edge

HAZARD_TTC_THRESHOLD_S = 2.0


def _clock_hour(bearing_rad: float) -> int:
    degrees = np.degrees(bearing_rad) % 360.0
    clock = round(degrees / 30.0) % 12
    return clock if clock != 0 else 12


def describe(concept: str, edge: Edge) -> str:
    clock = _clock_hour(edge.bearing_rad)
    if edge.ttc_s is not None and edge.ttc_s < HAZARD_TTC_THRESHOLD_S:
        return f"{concept}, {clock} o'clock, {edge.ttc_s:.0f}s"
    return f"{concept} at {clock} o'clock"
