"""What is safe to remember: a concept's motion class.

The memory graph's premise -- a node keeps its last world position while
out of view -- is right for a pole and wrong for a car: a remembered
moving object is a guess that gets worse every frame it isn't
re-observed. Persistence and velocity handling depend on which class a
concept falls into:

- static (pole, tree, bin, ...): velocity is pinned to zero rather than
  estimated from noisy per-frame lifting.
- movable (person, car, ...): velocity is estimated, but a sample faster
  than the class could physically move is treated as an association or
  lifting jump, not motion, and the node goes stale fast once unobserved
  -- see SpatialMemoryGraph.
- unknown: anything not listed keeps the original, class-agnostic
  behaviour, which carries none of the above protection. The hardcoded
  vocabulary can't be exhaustive (Spark's proposed concepts are free
  text), so `register_motion_classification` lets a concept be
  classified once it's actually seen, by asking Spark the same question
  this file answers for free -- see
  locus.perception.spark_proposer.classify_motion.

Matching is on whole words of the concept string, so "street lamp",
"trash can" and "parked van" all resolve; movable wins a tie ("parked
car" is still a car). The hardcoded lists are checked first and always
win over a registered classification, so a Spark call can only add
caution to a concept, never remove it from an already-verified one.
"""

from __future__ import annotations

import re

STATIC = "static"
MOVABLE = "movable"
UNKNOWN = "unknown"

# A classification registered for a concept the hardcoded lists below
# don't cover, keyed on the exact concept string as proposed/segmented
# (see register_motion_classification). Starts empty; untouched, every
# function here behaves exactly as it did before this existed.
_registered: dict[str, tuple[str, float | None]] = {}

_STATIC_WORDS = {
    "pole", "post", "lamppost", "lamp", "light", "streetlight", "tree", "trunk", "hydrant",
    "bin", "trashcan", "dumpster", "sign", "signpost", "bench", "bollard", "pillar", "column",
    "fence", "wall", "railing", "barrier", "cone", "planter", "mailbox", "meter", "kiosk",
    "shelter", "stair", "step", "curb", "kerb", "building", "door", "hedge", "bush",
    "chair", "table", "couch", "sofa", "shelf", "cabinet", "counter", "bed", "desk",
}
_STATIC_PHRASES = {"trash can", "garbage can", "fire hydrant", "traffic light", "street light", "parking meter"}

# Fallback ceiling for an unlisted concept when Spark classification itself
# fails (no API key, network, rate limit) -- see generate_dashboard_data.py's
# classify_unlisted_concepts. Deliberately tight (person-level, not
# vehicle-level): the point is to not ship fully unprotected, not to guess
# the concept's real speed.
DEFAULT_CAUTIOUS_MAX_SPEED_MPS = 3.0

# Generous physical ceilings (m/s): only there to reject impossible samples.
_MAX_SPEED_MPS = {
    "person": 3.0, "pedestrian": 3.0, "people": 3.0, "man": 3.0, "woman": 3.0, "child": 3.0,
    "kid": 3.0, "jogger": 4.0, "runner": 4.0, "stroller": 3.0, "wheelchair": 3.0,
    "dog": 6.0,
    "bicycle": 10.0, "bike": 10.0, "cyclist": 10.0, "scooter": 10.0, "skateboard": 8.0,
    "motorcycle": 20.0, "motorbike": 20.0, "car": 20.0, "taxi": 20.0, "cab": 20.0, "van": 20.0,
    "truck": 20.0, "lorry": 20.0, "bus": 20.0, "vehicle": 20.0, "suv": 20.0, "tram": 20.0,
}


def _words(concept: str) -> list[str]:
    return re.findall(r"[a-z]+", concept.lower())


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


def motion_class(concept: str) -> str:
    words = [_singular(w) for w in _words(concept)]
    if any(w in _MAX_SPEED_MPS for w in words):
        return MOVABLE
    joined = " ".join(words)
    if any(w in _STATIC_WORDS for w in words) or any(p in joined for p in _STATIC_PHRASES):
        return STATIC
    if concept in _registered:
        return _registered[concept][0]
    return UNKNOWN


def max_speed_mps(concept: str) -> float | None:
    """Physical speed ceiling for a movable concept, None otherwise."""
    speeds = [_MAX_SPEED_MPS[w] for w in (_singular(w) for w in _words(concept)) if w in _MAX_SPEED_MPS]
    if speeds:
        return max(speeds)
    # motion_class() already checks the hardcoded static list before the
    # registry; reusing it here (rather than re-reading _registered
    # directly) is what makes "pole" ignore a registered override the same
    # way motion_class("pole") does -- the bug caught by
    # test_hardcoded_vocabulary_always_wins_over_a_registration.
    if motion_class(concept) != MOVABLE:
        return None
    registered = _registered.get(concept)
    return registered[1] if registered else None


def register_motion_classification(concept: str, kind: str, max_speed: float | None) -> None:
    """Record a classification for a concept the hardcoded vocabulary
    above doesn't cover, keyed on the exact string as it appears in the
    segmenter's concept list (the same string every Node/Instance3D for
    it will carry). Called once per unlisted concept per run -- see
    generate_dashboard_data.py's classify_unlisted_concepts. A concept
    the hardcoded lists already match ignores this entirely (checked
    first in motion_class()/max_speed_mps() above), so this can only add
    caution, never remove it."""
    assert kind in (STATIC, MOVABLE), kind
    _registered[concept] = (kind, max_speed if kind == MOVABLE else None)


def reset_registered_classifications() -> None:
    """Test isolation / a fresh run's unlisted-concept registry start."""
    _registered.clear()
