"""Edge -> short spoken directive. At most six words: this is heard while
walking, not read, so it has to resolve in about a second of TTS.

This is deliberately the fast, local, no-API-call path -- the one always
used for anything on a collision course (critical or approaching),
where a few seconds of margin don't exist to spend on a network round
trip, and the only fallback if the warmer, narrated path (locus.actuation.
narrator.SparkNarrator, used for everything with real margin) is
unavailable or fails."""

from __future__ import annotations

import numpy as np

from locus.types import Edge

HAZARD_TTC_THRESHOLD_S = 2.0
# Earlier "approaching" tier, the same 5s lead time dashboard/static/js/
# state.js calls HAZARD_TTC_S.
APPROACH_TTC_THRESHOLD_S = 5.0
# A closing object whose closest point of approach stays at least this far
# from the wearer's centre passes by rather than hits: ~0.3m shoulder
# half-width plus ~0.45m margin for heading error. Without it, walking
# past something a metre to the side reads as a collision course. Such
# objects are still tracked and spoken as ordinary "at N o'clock"
# mentions, just never as hazards.
CPA_GATE_M = 0.75


def clock_hour(bearing_rad: float) -> int:
    degrees = np.degrees(bearing_rad) % 360.0
    clock = round(degrees / 30.0) % 12
    return clock if clock != 0 else 12


def on_collision_course(edge: Edge) -> bool:
    return edge.ttc_s is not None and (edge.cpa_m is None or edge.cpa_m < CPA_GATE_M)


def is_critical(edge: Edge) -> bool:
    return on_collision_course(edge) and edge.ttc_s < HAZARD_TTC_THRESHOLD_S


def severity(edge: Edge) -> str:
    """"critical" / "approaching" / "advisory" -- mirrors state.js's
    hazardSeverity() ("hazard" there is "approaching" here)."""
    if is_critical(edge):
        return "critical"
    if on_collision_course(edge) and edge.ttc_s < APPROACH_TTC_THRESHOLD_S:
        return "approaching"
    return "advisory"


def describe(concept: str, edge: Edge) -> str:
    """Anything on a collision course -- critical or approaching -- gets
    this short form with its seconds. Speaking eats lead time: this phrase
    is ~1.6s of SeamlessM4T speech, the narrated sentence 2.2-2.6s, which
    an object 4s out can't spare."""
    clock = clock_hour(edge.bearing_rad)
    if severity(edge) != "advisory":
        return f"{concept}, {clock} o'clock, {edge.ttc_s:.0f}s"
    return f"{concept} at {clock} o'clock"
