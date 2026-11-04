"""Which single edge, if any, gets spoken this frame.

A blind listener hears this in real time, so two things are rationed:
the channel (one cue at a time, with silence between cues -- the
cooldowns) and attention (say each thing once, then only again when
something about it changed -- the novelty rule). What counts as news
about an object:
  - it has never been announced;
  - its severity went up (advisory -> approaching -> critical);
  - it moved two or more clock hours while in front of the wearer (a
    turn, or something crossing the path) -- not the ordinary 1 -> 3
    o'clock drift of walking past it, which is expected, not news.
Routine (advisory) mentions are limited to things ahead of or beside the
wearer; something behind only gets a word if it is actually approaching.
A critical edge is always a candidate regardless (re-confirmed every
CRITICAL_COOLDOWN_S), and an escalation above the severity of the last
thing said cuts through the cooldown (the dashboard mirrors this, letting
a higher-severity clip interrupt a lower one). Safety beats pacing;
routine updates don't.

Routine mentions must also be timely: an object only becomes ambient news
within ADVISORY_RANGE_M, and a mention still waiting behind the cooldown
after ADVISORY_MAX_WAIT_S is dropped rather than spoken late -- describing
something the wearer has already passed is worse than saying nothing.
Approaching and critical warnings are never dropped this way.

A rise in severity has to hold for ESCALATION_CONFIRM_S before it counts;
until then the edge keeps the highest severity it has held that long.
This filters out a single noisy frame reading as a real, sustained
approach -- the hold covers "approaching" too, not just critical, since
escalations otherwise cut through the cooldown on a one-frame blip just
as readily as on a real one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from locus.actuation.directives import clock_hour, severity
from locus.types import Edge

DIRECTIVE_COOLDOWN_S = 3.0  # long enough to finish a ~1s phrase and leave a breath after it
CRITICAL_COOLDOWN_S = 1.2  # a still-critical object gets re-confirmed more often, but still not every frame
CLOCK_CHANGE_HOURS = 2
FRONT_CLOCK_HOURS = (10, 11, 12, 1, 2)
BEHIND_CLOCK_HOURS = (4, 5, 6, 7, 8)
# How long an object must stay below its announced severity before that
# announcement is forgotten, so approaching it again later is news again --
# long enough that severity flicker on noisy TTC doesn't re-trigger it.
DEESCALATION_HOLD_S = 2.0
ESCALATION_CONFIRM_S = 0.13  # three consecutive frames at 15 fps
ADVISORY_RANGE_M = 10.0
ADVISORY_MAX_WAIT_S = 2.0

SEVERITY_RANK = {"advisory": 0, "approaching": 1, "critical": 2}


def most_urgent_edge(edges: list[Edge], severity_of=severity) -> Edge | None:
    """Highest severity first; among closing edges the soonest, among
    advisory ones the nearest. Mirrored by state.js's mostUrgentEdge()."""
    if not edges:
        return None

    def key(edge: Edge):
        sev = severity_of(edge)
        urgency = edge.ttc_s if sev != "advisory" else math.inf
        return (-SEVERITY_RANK[sev], urgency, edge.range_m)

    return min(edges, key=key)


def _clock_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


@dataclass
class _Announcement:
    severity: str
    clock: int
    below_since: float | None = None


class SpeechPolicy:
    def __init__(
        self,
        cooldown_s: float = DIRECTIVE_COOLDOWN_S,
        critical_cooldown_s: float = CRITICAL_COOLDOWN_S,
    ):
        self.cooldown_s = cooldown_s
        self.critical_cooldown_s = critical_cooldown_s
        self.last_spoken_t: float | None = None
        self.last_spoken_severity: str | None = None
        self._announced: dict[int, _Announcement] = {}
        self._at_least_since: dict[int, dict[str, float]] = {}  # node_id -> severity -> held continuously since
        self._news_since: dict[int, float] = {}  # node_id -> when it became (and stayed) news
        # This frame's severity per node after the critical hold -- what
        # the timeline should show, so the HUD agrees with the audio.
        self.severities: dict[int, str] = {}

    def _update_severities(self, edges: list[Edge], t: float) -> None:
        at_least_since = {}
        self.severities = {}
        for edge in edges:
            rank = SEVERITY_RANK[severity(edge)]
            previous = self._at_least_since.get(edge.dst_node_id, {})
            held = at_least_since[edge.dst_node_id] = {
                level: previous.get(level, t) for level, r in SEVERITY_RANK.items() if 0 < r <= rank
            }
            confirmed = [level for level, since in held.items() if t - since >= ESCALATION_CONFIRM_S - 1e-9]
            self.severities[edge.dst_node_id] = max(confirmed, key=SEVERITY_RANK.get, default="advisory")
        self._at_least_since = at_least_since

    def _severity(self, edge: Edge) -> str:
        return self.severities[edge.dst_node_id]

    def _is_news(self, edge: Edge, t: float) -> bool:
        prev = self._announced.get(edge.dst_node_id)
        if prev is None:
            return True
        sev, clock = self._severity(edge), clock_hour(edge.bearing_rad)
        if SEVERITY_RANK[sev] < SEVERITY_RANK[prev.severity]:
            if prev.below_since is None:
                prev.below_since = t
            elif t - prev.below_since >= DEESCALATION_HOLD_S:
                prev.severity, prev.below_since = sev, None
            return False
        prev.below_since = None
        if SEVERITY_RANK[sev] > SEVERITY_RANK[prev.severity]:
            return True
        return clock in FRONT_CLOCK_HOURS and _clock_distance(clock, prev.clock) >= CLOCK_CHANGE_HOURS

    def select(self, edges: list[Edge], t: float) -> int | None:
        """The node id to speak about at time `t`, or None for silence."""
        self._update_severities(edges, t)
        candidates = []
        for e in edges:
            advisory = self._severity(e) == "advisory"
            if advisory and e.range_m > ADVISORY_RANGE_M:
                continue
            news = self._is_news(e, t)
            if not news:
                self._news_since.pop(e.dst_node_id, None)
            if not (news or self._severity(e) == "critical"):
                continue
            if advisory and clock_hour(e.bearing_rad) in BEHIND_CLOCK_HOURS:
                continue
            since = self._news_since.setdefault(e.dst_node_id, t)
            if advisory and t - since > ADVISORY_MAX_WAIT_S:
                # Too late to be useful: retire it as if said, so it only
                # comes back on a real change (escalation, a turn).
                self._mark_announced(e)
                continue
            candidates.append(e)
        chosen = most_urgent_edge(candidates, self._severity)
        if chosen is None:
            return None

        sev = self._severity(chosen)
        elapsed = math.inf if self.last_spoken_t is None else t - self.last_spoken_t
        cooldown = self.critical_cooldown_s if sev == "critical" else self.cooldown_s
        escalating = sev != "advisory" and (
            self.last_spoken_severity is None or SEVERITY_RANK[sev] > SEVERITY_RANK[self.last_spoken_severity]
        )
        if not (escalating or elapsed >= cooldown):
            return None

        self.last_spoken_t = t
        self.last_spoken_severity = sev
        self._mark_announced(chosen)
        return chosen.dst_node_id

    def _mark_announced(self, edge: Edge) -> None:
        self._announced[edge.dst_node_id] = _Announcement(self._severity(edge), clock_hour(edge.bearing_rad))
        self._news_since.pop(edge.dst_node_id, None)
