"""Scoring a condition's answer against ground truth.

Bearing is scored on the clock face (hours), the same unit the audio UX
and the questions it's answering are both expressed in -- a clockface
circular difference, not a linear one, since 11 o'clock and 1 o'clock are
two hours apart, not ten.
"""

from __future__ import annotations

import numpy as np

from locus.eval.conditions import ConditionAnswer


def clockface_error(predicted_clock: float, true_clock: float) -> float:
    """Signed circular difference between two clock positions, in hours,
    wrapped to (-6, 6]."""
    diff = (predicted_clock - true_clock + 6) % 12 - 6
    return diff


def evaluate_answer(answer: ConditionAnswer, true_bearing_clock: float, true_range_m: float) -> dict:
    if not answer.known:
        return {"answerable": False, "bearing_error_hours": None, "range_error_m": None}
    return {
        "answerable": True,
        "bearing_error_hours": abs(clockface_error(answer.bearing_clock, true_bearing_clock)),
        "range_error_m": abs(answer.range_m - true_range_m) if answer.range_m is not None else None,
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {
            "n_queries": 0,
            "answer_rate": 0.0,
            "mean_bearing_error_hours": None,
            "mean_range_error_m": None,
        }
    answerable = [r for r in results if r["answerable"]]
    bearing_errors = [r["bearing_error_hours"] for r in answerable]
    range_errors = [r["range_error_m"] for r in answerable if r["range_error_m"] is not None]
    return {
        "n_queries": len(results),
        "answer_rate": len(answerable) / len(results),
        "mean_bearing_error_hours": float(np.mean(bearing_errors)) if bearing_errors else None,
        "mean_range_error_m": float(np.mean(range_errors)) if range_errors else None,
    }
