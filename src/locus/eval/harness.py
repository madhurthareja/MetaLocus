"""CLI entry point for the A/B/C/D ablation: `python -m locus.eval.harness`.

Runs every synthetic scenario through every condition, splits scored
questions by ground-truth visibility (in-view vs out-of-view), and prints
per-condition accuracy. The out-of-view columns are the whole point:
that's where C/D (persistent memory) should separate from A/B (no/short
memory), and where C should separate from D (box vs mask lifting noise).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from locus.data import synthetic
from locus.eval.conditions import ConditionA, ConditionB, ConditionC, ConditionD
from locus.eval.metrics import aggregate, evaluate_answer
from locus.eval.questions import Question, generate_questions
from locus.perception.mock_segmenter import MockSegmenter

SCENARIOS = {
    "straight_walk": synthetic.straight_walk,
    "head_turn_away": synthetic.head_turn_away,
    "head_turn_return": synthetic.head_turn_return,
    "cluttered_occlusion": synthetic.cluttered_occlusion,
    "crossing_hazard": synthetic.crossing_hazard,
}

CONDITIONS = {
    "A_stateless": lambda intr: ConditionA(intr),
    "B_window": lambda intr: ConditionB(intr, window_s=1.0),
    "C_box": lambda intr: ConditionC(intr),
    "D_mask": lambda intr: ConditionD(intr),
}

app = typer.Typer()


def run_condition(condition, frames, ground_truth, questions: list[Question]) -> dict:
    segmenter = MockSegmenter(ground_truth)
    concepts = sorted({obj.concept for obj in ground_truth.objects})
    segmenter.start(concepts)
    condition.reset()

    questions_by_frame: dict[int, list[Question]] = {}
    for q in questions:
        questions_by_frame.setdefault(q.frame_index, []).append(q)

    in_view_results, out_of_view_results = [], []
    for i, frame in enumerate(frames):
        detections = segmenter.step(frame)
        condition.observe(frame, detections)
        for q in questions_by_frame.get(i, []):
            answer = condition.answer(q.concept)
            scored = evaluate_answer(answer, q.true_bearing_clock, q.true_range_m)
            (in_view_results if q.in_view else out_of_view_results).append(scored)

    return {"in_view": aggregate(in_view_results), "out_of_view": aggregate(out_of_view_results)}


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


@app.command()
def run(stride: int = 1) -> None:
    intrinsics = synthetic.DEFAULT_INTRINSICS
    console = Console()

    for scenario_name, scenario_fn in SCENARIOS.items():
        frames, ground_truth = scenario_fn()
        questions = generate_questions(ground_truth, stride=stride)

        table = Table(title=scenario_name)
        table.add_column("condition")
        table.add_column("in-view answer rate")
        table.add_column("in-view bearing err (hr)")
        table.add_column("out-of-view answer rate")
        table.add_column("out-of-view bearing err (hr)")

        for condition_name, make_condition in CONDITIONS.items():
            condition = make_condition(intrinsics)
            result = run_condition(condition, frames, ground_truth, questions)
            iv, oov = result["in_view"], result["out_of_view"]
            table.add_row(
                condition_name,
                _fmt(iv["answer_rate"]),
                _fmt(iv["mean_bearing_error_hours"]),
                _fmt(oov["answer_rate"]),
                _fmt(oov["mean_bearing_error_hours"]),
            )
        console.print(table)


if __name__ == "__main__":
    app()
