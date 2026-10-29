"""Detection (2D) -> Instance3D (metric, camera frame).

Median depth over the region, not mean: mask/box boundaries sit on
background pixels, and the mean gets dragged toward them. This is
exactly why masks should beat boxes on cluttered/occluded scenes -- the
mask excludes background pixels a box can't, so its median is less
contaminated. `lift` and `lift_from_box` share every line except which
pixels they take the median over, so the A/B/C/D ablation's C-vs-D
comparison is controlled: same code path, only the region differs.
"""

from __future__ import annotations

import numpy as np

from locus.geometry.intrinsics import Intrinsics
from locus.types import Detection, Instance3D

MIN_VALID_DEPTH_PIXELS = 20


def _lift_region(
    detection: Detection,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    region_mask: np.ndarray,
    min_valid_pixels: int,
) -> Instance3D | None:
    valid = region_mask & (depth > 0) & np.isfinite(depth)
    if valid.sum() < min_valid_pixels:
        return None

    median_depth = float(np.median(depth[valid]))
    ys, xs = np.nonzero(region_mask)
    centroid_px = np.array([xs.mean(), ys.mean()])
    p_cam = intrinsics.backproject(centroid_px, median_depth)

    region_xs = xs * median_depth / intrinsics.fx
    region_ys = ys * median_depth / intrinsics.fy
    extent = np.array(
        [
            np.ptp(region_xs) if len(xs) > 1 else 0.0,
            np.ptp(region_ys) if len(ys) > 1 else 0.0,
            0.0,
        ]
    )

    return Instance3D(
        concept=detection.concept,
        instance_id=detection.instance_id,
        p_cam=p_cam,
        p_world=p_cam,  # caller transforms to world via geometry.frames.ego_to_world
        extent=extent,
        score=detection.score,
    )


def lift(
    detection: Detection,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    min_valid_pixels: int = MIN_VALID_DEPTH_PIXELS,
) -> Instance3D | None:
    """Condition D: median depth over the mask."""
    return _lift_region(detection, depth, intrinsics, detection.mask, min_valid_pixels)


def lift_from_box(
    detection: Detection,
    depth: np.ndarray,
    intrinsics: Intrinsics,
    min_valid_pixels: int = MIN_VALID_DEPTH_PIXELS,
) -> Instance3D | None:
    """Condition C: median depth over the bounding box (masks discarded)."""
    h, w = depth.shape
    bx, by, bw, bh = detection.box_xywh_norm
    x0, y0 = int(bx * w), int(by * h)
    x1, y1 = int((bx + bw) * w), int((by + bh) * h)
    box_mask = np.zeros_like(depth, dtype=bool)
    box_mask[y0:y1, x0:x1] = True
    return _lift_region(detection, depth, intrinsics, box_mask, min_valid_pixels)
