"""A Segmenter that reads ground truth instead of running SAM -- this is
what lets the eval harness compare conditions A/B/C/D without ever needing
a GPU or model weights. It implements locus.perception.base.Segmenter
exactly, so a real backend is a drop-in replacement for it everywhere else.

`instability` simulates a segmenter losing track of an object for a frame
(a false negative + an instance_id reassignment on the next sighting),
which is what exercises SpatialMemoryGraph's spatial re-association
fallback (_find_spatial_match) instead of its instance_id fast path.
`position_noise_px` jitters the returned mask/box by a few pixels, which
is what the C-vs-D (box vs mask lifting) comparison needs noise for --
without it, both lifting functions would be measuring a geometry they'd
nail exactly, and the comparison would be uninteresting.
"""

from __future__ import annotations

import numpy as np

from locus.data.synthetic import GroundTruth
from locus.types import Detection, Frame


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Zero-padded integer-pixel shift (no wraparound)."""
    h, w = mask.shape
    shifted = np.zeros_like(mask)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _shift_box(
    box_xywh_norm: tuple[float, float, float, float], dx_frac: float, dy_frac: float
) -> tuple[float, float, float, float]:
    x, y, w, h = box_xywh_norm
    x = float(np.clip(x + dx_frac, 0.0, max(0.0, 1.0 - w)))
    y = float(np.clip(y + dy_frac, 0.0, max(0.0, 1.0 - h)))
    return (x, y, w, h)


class MockSegmenter:
    def __init__(
        self,
        ground_truth: GroundTruth,
        instability: float = 0.0,
        position_noise_px: float = 0.0,
        seed: int = 0,
    ):
        self._ground_truth = ground_truth
        self._instability = instability
        self._position_noise_px = position_noise_px
        self._rng = np.random.default_rng(seed)

        self._active_concepts: set[str] = set()
        self._instance_ids: dict[str, int] = {}  # object_id -> instance_id
        self._next_instance_id = 1
        self._frame_index = 0

    def start(self, concepts: list[str]) -> None:
        self._active_concepts = set(concepts)
        self._frame_index = 0

    def update_concepts(self, concepts: list[str]) -> None:
        self._active_concepts = set(concepts)

    def step(self, frame: Frame) -> list[Detection]:
        if self._frame_index >= len(self._ground_truth.frames):
            raise IndexError("MockSegmenter stepped past the end of its ground-truth sequence")
        gt_frame = self._ground_truth.frames[self._frame_index]
        self._frame_index += 1

        detections = []
        for obj in self._ground_truth.objects:
            if obj.concept not in self._active_concepts:
                continue
            if not gt_frame.object_visible.get(obj.object_id, False):
                self._instance_ids.pop(obj.object_id, None)
                continue
            if self._instability > 0 and self._rng.random() < self._instability:
                self._instance_ids.pop(obj.object_id, None)  # dropped frame: track lost
                continue

            mask = gt_frame.object_mask.get(obj.object_id)
            box = gt_frame.object_box_norm.get(obj.object_id)
            if mask is None or box is None:
                continue

            if self._position_noise_px > 0:
                dy, dx = self._rng.normal(0, self._position_noise_px, size=2)
                mask = _shift_mask(mask, int(round(dy)), int(round(dx)))
                h, w = mask.shape
                box = _shift_box(box, dx / w, dy / h)

            instance_id = self._instance_ids.get(obj.object_id)
            if instance_id is None:
                instance_id = self._next_instance_id
                self._next_instance_id += 1
                self._instance_ids[obj.object_id] = instance_id

            detections.append(
                Detection(
                    concept=obj.concept,
                    instance_id=instance_id,
                    mask=mask,
                    box_xywh_norm=box,
                    score=1.0,
                )
            )
        return detections

    def close(self) -> None:
        pass
