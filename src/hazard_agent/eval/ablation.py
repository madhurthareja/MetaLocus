"""A/B/C/D ablation ladder.

A - Spark 1.3 on current frame only (stateless captioner)
B - Spark 1.3 on a sliding window of the last N frames
C - Spark + SAM boxes only, masks discarded (free from the same detector pass)
D - Full system: masks + persistent 3D graph

C vs D is where the thesis lives or dies -- report it honestly even if
C ~= D on a given sequence. See docs/PROJECT_PLAN.md section 4.
"""

from enum import Enum


class AblationCondition(str, Enum):
    A_STATELESS = "A"
    B_SLIDING_WINDOW = "B"
    C_BOXES_ONLY = "C"
    D_FULL_GRAPH = "D"


def run_condition(condition: AblationCondition, sequence, questions):
    raise NotImplementedError
