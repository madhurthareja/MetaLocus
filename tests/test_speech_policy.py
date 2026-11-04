import numpy as np
import pytest

from locus.actuation.speech_policy import CRITICAL_COOLDOWN_S, DIRECTIVE_COOLDOWN_S, SpeechPolicy, most_urgent_edge
from locus.types import Edge


def _edge(node_id, range_m=3.0, bearing_deg=0.0, ttc_s=None, cpa_m=None):
    return Edge(src_node_id=0, dst_node_id=node_id, range_m=range_m, bearing_rad=np.radians(bearing_deg),
                elevation_delta_m=0.0, ttc_s=ttc_s, cpa_m=cpa_m)


def _run(policy, frames, dt=1 / 15):
    out = []
    for i, edges in enumerate(frames):
        node = policy.select(edges, i * dt)
        if node is not None:
            out.append((i * dt, node))
    return out


def test_an_unchanged_object_is_announced_once_not_every_cooldown():
    """The old rule re-announced the nearest object every 3s for as long as
    it stayed nearest -- on the real SANPO walk, the same pole, repeatedly."""
    frames = [[_edge(1, bearing_deg=30)]] * 150  # 10s of the same pole
    assert [node for _, node in _run(SpeechPolicy(), frames)] == [1]


def test_a_new_object_is_announced_after_the_cooldown():
    frames = [[_edge(1)]] * 35 + [[_edge(1), _edge(2, range_m=5.0)]] * 60
    spoken = _run(SpeechPolicy(), frames)
    assert [node for _, node in spoken] == [1, 2]
    assert spoken[1][0] >= DIRECTIVE_COOLDOWN_S - 1e-9, "the channel still gets its breath between cues"


def test_a_routine_mention_that_waited_too_long_is_dropped_not_spoken_late():
    """Objects entering range together used to queue behind the cooldown
    and be described seconds after the wearer had seen (or passed) them."""
    frames = [[_edge(1)]] * 10 + [[_edge(1), _edge(2, range_m=5.0)]] * 60
    spoken = _run(SpeechPolicy(), frames)
    assert [node for _, node in spoken] == [1]


def test_routine_mentions_wait_until_the_object_is_near():
    frames = [[_edge(1, range_m=15.0)]] * 30 + [[_edge(1, range_m=8.0)]] * 10
    spoken = _run(SpeechPolicy(), frames)
    assert [round(t, 2) for t, _ in spoken] == [2.0]


def test_a_late_warning_is_never_dropped():
    frames = [[_edge(1)]] * 10 + [[_edge(1), _edge(2, range_m=12.0, ttc_s=4.0, cpa_m=0.2)]] * 60
    spoken = _run(SpeechPolicy(), frames)
    assert [node for _, node in spoken] == [1, 2]


def test_escalation_to_critical_cuts_in_and_is_reconfirmed():
    frames = [[_edge(1, ttc_s=4.0, cpa_m=0.2)]] * 5 + [[_edge(1, range_m=1.5, ttc_s=1.0, cpa_m=0.2)]] * 40
    spoken = _run(SpeechPolicy(), frames)
    times = [t for t, _ in spoken]
    assert times[0] == 0.0, "first mentioned at once, before any severity has held"
    assert times[1] == pytest.approx(2 / 15), "approaching cuts in once it has held 3 frames"
    assert times[2] == pytest.approx(7 / 15), "critical too: 3 frames after it starts, not after the cooldown"
    gaps = np.diff(times[2:])
    assert len(gaps) >= 1 and np.all(gaps >= CRITICAL_COOLDOWN_S - 1e-9)


def test_walking_past_an_object_is_not_news_but_a_turn_is():
    passing = [[_edge(1, bearing_deg=b)] for b in np.linspace(20, 100, 60)]  # 1 -> 3 o'clock as it goes by
    assert len(_run(SpeechPolicy(), passing)) == 1
    turned = [[_edge(1, bearing_deg=60)]] * 50 + [[_edge(1, bearing_deg=-30)]] * 10  # 2 -> 11 o'clock
    assert len(_run(SpeechPolicy(), turned)) == 2


def test_most_urgent_prefers_a_real_collision_course_over_a_nearer_passing_object():
    passing_pole = _edge(1, range_m=1.5, ttc_s=1.2, cpa_m=1.0)
    approaching_bike = _edge(2, range_m=8.0, ttc_s=3.5, cpa_m=0.3)
    assert most_urgent_edge([passing_pole, approaching_bike]).dst_node_id == 2


def test_routine_mentions_skip_things_behind_but_not_things_approaching_from_behind():
    assert _run(SpeechPolicy(), [[_edge(1, bearing_deg=180)]] * 30) == []
    cyclist = [[_edge(1, range_m=6.0, bearing_deg=170, ttc_s=3.0, cpa_m=0.3)]] * 5
    assert [node for _, node in _run(SpeechPolicy(), cyclist)] == [1]


def test_a_one_frame_critical_blip_is_not_spoken_as_critical():
    """Real SANPO: a parked van's lifted position slid toward the wearer for
    exactly one frame as it was walked past. A real collision course stays
    critical; a blip reads as approaching and never cuts in as critical."""
    blip = [[_edge(1, ttc_s=6.0, cpa_m=0.3)]] * 3 + [[_edge(1, range_m=3.0, ttc_s=1.3, cpa_m=0.3)]] + [[_edge(1, ttc_s=6.0, cpa_m=0.3)]] * 30
    policy = SpeechPolicy()
    severities = []
    for i, edges in enumerate(blip):
        policy.select(edges, i / 15)
        severities.append(policy.severities[1])
    assert "critical" not in severities

    sustained = [[_edge(1, range_m=3.0, ttc_s=1.3, cpa_m=0.3)]] * 10
    spoken = _run(SpeechPolicy(), sustained)
    assert spoken[0][0] == 0.0, "still announced at once, as approaching"
    assert any(t == pytest.approx(2 / 15) for t, _ in spoken), "and cuts in as critical once it has held 3 frames"


def test_an_ambient_mention_does_not_hold_back_an_approaching_warning():
    """synthetic_crossing_hazard: the bicycle was mentioned at t=0, then its
    'approaching' escalation sat behind the 3s cooldown until the clip
    ended. Escalating past what was last said cuts in."""
    frames = [[_edge(1, range_m=8.0, ttc_s=8.0, cpa_m=0.1)]] * 15 + [[_edge(1, range_m=7.0, ttc_s=4.5, cpa_m=0.1)]] * 30
    spoken = _run(SpeechPolicy(), frames)
    assert [round(t, 2) for t, _ in spoken] == [0.0, round(17 / 15, 2)], "the warning waits only for its 3-frame hold"
