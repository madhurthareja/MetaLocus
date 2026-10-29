"""Query generation for the eval harness.

One question per (sampled frame, object): "where is <concept> right now".
Each question is tagged with ground-truth visibility at query time, so
results can be split into in-view vs out-of-view accuracy -- "where is X"
asked while X is out of view is the one question type the whole benchmark
thesis is about (persistent world-frame memory vs. frame-level captioning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from locus.data.synthetic import GroundTruth


@dataclass(frozen=True)
class Question:
    frame_index: int
    t: float
    concept: str
    object_id: str
    in_view: bool  # ground-truth visibility at query time
    true_bearing_clock: float
    true_range_m: float


def generate_questions(ground_truth: GroundTruth, stride: int = 1) -> list[Question]:
    """`stride` subsamples frames (every Nth) to keep long sequences cheap."""
    questions = []
    for gt_frame in ground_truth.frames[::stride]:
        wearer_pos = gt_frame.pose.t_world_cam
        for obj in ground_truth.objects:
            p_world = gt_frame.object_position_world[obj.object_id]
            questions.append(
                Question(
                    frame_index=gt_frame.frame_index,
                    t=gt_frame.t,
                    concept=obj.concept,
                    object_id=obj.object_id,
                    in_view=gt_frame.object_visible.get(obj.object_id, False),
                    true_bearing_clock=gt_frame.object_bearing_body[obj.object_id],
                    true_range_m=float(np.linalg.norm(p_world - wearer_pos)),
                )
            )
    return questions
