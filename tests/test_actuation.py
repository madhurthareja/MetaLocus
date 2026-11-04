import numpy as np
import pytest

from locus.actuation.audio import render_cue
from locus.actuation.directives import describe
from locus.types import Edge


def _edge(bearing_rad: float = 0.0, ttc_s: float | None = None, range_m: float = 3.0) -> Edge:
    return Edge(
        src_node_id=0, dst_node_id=1, range_m=range_m,
        bearing_rad=bearing_rad, elevation_delta_m=0.0, ttc_s=ttc_s,
    )


@pytest.mark.parametrize(
    "ttc_s",
    [None, 5.0, 1.5, 0.4],
)
def test_describe_never_exceeds_six_words(ttc_s):
    text = describe("chair", _edge(ttc_s=ttc_s))
    assert len(text.split()) <= 6


def test_describe_flags_imminent_hazard_but_not_distant_one():
    near = describe("bicycle", _edge(ttc_s=0.5))
    far = describe("bicycle", _edge(ttc_s=10.0))
    assert "," in near and near.split(",")[-1].strip().endswith("s"), "near hazard should state seconds-to-collision"
    assert "," not in far, "distant object shouldn't be framed as a hazard"


def test_cue_pans_toward_bearing():
    left = render_cue(_edge(bearing_rad=-np.pi / 4))
    right = render_cue(_edge(bearing_rad=np.pi / 4))
    center = render_cue(_edge(bearing_rad=0.0))

    assert np.abs(left[:, 0]).mean() > np.abs(left[:, 1]).mean(), "negative bearing should favor the left channel"
    assert np.abs(right[:, 1]).mean() > np.abs(right[:, 0]).mean(), "positive bearing should favor the right channel"
    assert np.abs(center[:, 0]).mean() == pytest.approx(np.abs(center[:, 1]).mean(), abs=1e-6)


def _envelope_variance(cue: np.ndarray, n_chunks: int = 100) -> float:
    """Chunked-RMS envelope estimate: each chunk averages over many carrier
    cycles (so the 440Hz tone itself washes out) but is far shorter than a
    pulse period (so real amplitude modulation survives)."""
    mono = np.abs(cue[:, 0] + cue[:, 1])
    chunks = np.array_split(mono, n_chunks)
    chunk_means = [c.mean() for c in chunks]
    return float(np.var(chunk_means))


def test_cue_pulses_faster_as_ttc_shrinks():
    far_cue = render_cue(_edge(ttc_s=5.0), duration_s=1.0)
    near_cue = render_cue(_edge(ttc_s=0.5), duration_s=1.0)
    steady_cue = render_cue(_edge(ttc_s=None), duration_s=1.0)

    assert _envelope_variance(near_cue) > _envelope_variance(far_cue), "closer hazard should pulse harder/faster"
    assert _envelope_variance(steady_cue) < _envelope_variance(far_cue), "no TTC should be the steadiest tone"


def test_closing_object_that_passes_clear_is_not_critical():
    """Walking past a pole a metre to the side closes on it fast, but never
    reaches it -- measured on real SANPO, every remaining "critical" pole
    alert had a 1.0m miss distance."""
    from locus.actuation.directives import CPA_GATE_M, is_critical, severity

    passing = Edge(src_node_id=0, dst_node_id=1, range_m=1.7, bearing_rad=0.6, elevation_delta_m=0.0,
                   ttc_s=1.7, cpa_m=1.0)
    head_on = Edge(src_node_id=0, dst_node_id=1, range_m=1.7, bearing_rad=0.0, elevation_delta_m=0.0,
                   ttc_s=1.7, cpa_m=0.1)
    assert CPA_GATE_M < passing.cpa_m
    assert not is_critical(passing) and severity(passing) == "advisory"
    assert "," not in describe("pole", passing), "a passing object is mentioned, not framed as a hazard"
    assert is_critical(head_on) and severity(head_on) == "critical"


def test_approaching_object_gets_the_short_form_with_seconds():
    text = describe("bicycle", _edge(ttc_s=4.0))
    assert text == "bicycle, 12 o'clock, 4s"
