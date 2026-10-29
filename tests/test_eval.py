"""Regression test for the ablation harness itself: if this fails, the
harness has stopped demonstrating the project's actual thesis (persistent
world-frame memory beats frame-level captioning on out-of-view spatial
questions), whatever its individual pieces' unit tests say."""

from locus.data.synthetic import DEFAULT_INTRINSICS, head_turn_away
from locus.eval.conditions import ConditionA, ConditionD
from locus.eval.harness import run_condition
from locus.eval.questions import generate_questions


def test_persistent_memory_beats_stateless_on_out_of_view_questions():
    frames, ground_truth = head_turn_away(yaw_deg=60.0, hold_s=4.0)
    questions = generate_questions(ground_truth)

    result_a = run_condition(ConditionA(DEFAULT_INTRINSICS), frames, ground_truth, questions)
    result_d = run_condition(ConditionD(DEFAULT_INTRINSICS), frames, ground_truth, questions)

    assert result_a["out_of_view"]["answer_rate"] == 0.0, (
        "condition A has no persistence -- it should never answer an out-of-view question"
    )
    assert result_d["out_of_view"]["answer_rate"] > 0.9, (
        "condition D should answer nearly every out-of-view question via its decaying node"
    )

    # Both should be near-perfect while the object is actually in view --
    # the two conditions should differ on persistence, not on basic accuracy.
    assert result_a["in_view"]["answer_rate"] > 0.9
    assert result_d["in_view"]["answer_rate"] > 0.9
    assert result_d["out_of_view"]["mean_bearing_error_hours"] < 0.5
