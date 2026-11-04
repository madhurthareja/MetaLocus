import pytest

from locus.memory.motion_classes import (
    DEFAULT_CAUTIOUS_MAX_SPEED_MPS,
    MOVABLE,
    STATIC,
    UNKNOWN,
    max_speed_mps,
    motion_class,
    register_motion_classification,
    reset_registered_classifications,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registered_classifications()
    yield
    reset_registered_classifications()


def test_a_concept_outside_the_hardcoded_vocabulary_defaults_to_unknown():
    """The real failure this file exists to close: Spark proposed
    "suitcase" on a real SANPO walk, it matched neither word list, and its
    unprotected velocity swung +8 -> -17.9 m/s between consecutive frames
    with no suitcase visible in the frame. Confirms the starting point."""
    assert motion_class("suitcase") == UNKNOWN
    assert max_speed_mps("suitcase") is None


def test_registering_a_classification_closes_the_gap():
    register_motion_classification("suitcase", MOVABLE, 3.0)
    assert motion_class("suitcase") == MOVABLE
    assert max_speed_mps("suitcase") == 3.0


def test_registering_static_pins_no_speed():
    register_motion_classification("fire escape", STATIC, None)
    assert motion_class("fire escape") == STATIC
    assert max_speed_mps("fire escape") is None


def test_hardcoded_vocabulary_always_wins_over_a_registration():
    """A Spark call can only ever add caution, never remove it -- even a
    (nonsensical) registration for a known word must not override the
    hardcoded, already-verified classification."""
    register_motion_classification("car", STATIC, None)
    assert motion_class("car") == MOVABLE
    assert max_speed_mps("car") == 20.0

    register_motion_classification("pole", MOVABLE, 50.0)
    assert motion_class("pole") == STATIC
    assert max_speed_mps("pole") is None


def test_unregistered_unknown_concept_is_unaffected_by_other_registrations():
    """"puddle" is the existing class-agnostic-behavior test's example --
    registering something else must not accidentally cover it too."""
    register_motion_classification("suitcase", MOVABLE, 3.0)
    assert motion_class("puddle") == UNKNOWN


def test_default_cautious_speed_would_have_rejected_the_real_suitcase_noise():
    """The actual noise measured on real data: velocity swung as far as
    -17.9 m/s between frames. Confirms the fallback ceiling used when
    Spark classification itself fails is tight enough to reject it."""
    assert DEFAULT_CAUTIOUS_MAX_SPEED_MPS < 17.9
