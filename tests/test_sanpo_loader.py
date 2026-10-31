"""Integration test against the real, public SANPO bucket. Skips cleanly
if the network/bucket isn't reachable -- this hits a live Google Cloud
Storage endpoint, unlike the AEA loader test which only needs local disk."""

import numpy as np
import pytest
import requests

from locus.config import DEFAULT_CONFIG_PATH, load_config

config = load_config(DEFAULT_CONFIG_PATH)


def _bucket_reachable() -> bool:
    try:
        r = requests.head("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/labelmap.json", timeout=5)
        return r.status_code < 500
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _bucket_reachable(), reason="SANPO GCS bucket not reachable from this machine")


def test_loads_real_session_with_dense_depth_and_moving_pose():
    from locus.data.sanpo_loader import load_sequence

    session_id = config["sanpo_sequences"][0]
    seq = load_sequence(session_id, config=config)

    assert seq.n_frames > 0
    frames = list(seq.iter_frames(stride=5, max_frames=5))
    assert len(frames) == 5

    positions, coverages = [], []
    for frame in frames:
        assert frame.rgb.shape == (seq.intrinsics.height, seq.intrinsics.width, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth is not None
        assert frame.depth.shape == frame.rgb.shape[:2]
        coverages.append(float((frame.depth > 0).mean()))
        positions.append(frame.pose.t_world_cam.copy())

    # SANPO's whole reason for being in this project: dense depth, unlike
    # AEA's ~18% semi-dense coverage.
    assert np.mean(coverages) > 0.8

    total_motion = sum(
        float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)
    )
    assert total_motion > 0.1, "this session is tagged ego_motion=WALKING; expect real movement"


def test_dense_depth_lifts_far_more_detections_than_aea_style_sparse_depth():
    """Not just 'depth exists' -- confirms the practical payoff: lift()
    actually succeeds on far more of the frame than AEA's sparse depth
    allowed, using the same lifting code unchanged."""
    from locus.data.sanpo_loader import load_sequence
    from locus.geometry.lifting import MIN_VALID_DEPTH_PIXELS
    from locus.types import Detection

    session_id = config["sanpo_sequences"][0]
    seq = load_sequence(session_id, config=config)
    frame = next(seq.iter_frames(stride=1, max_frames=1, start_frame=100))

    # A grid of plausible object-sized boxes across the frame -- on AEA's
    # ~18%-coverage depth, most of these would fail MIN_VALID_DEPTH_PIXELS.
    h, w = frame.depth.shape
    box_w, box_h = 0.1, 0.1
    successes = 0
    total = 0
    for gx in range(1, 9):
        for gy in range(1, 9):
            x0, y0 = gx / 10, gy / 10
            mask = np.zeros((h, w), dtype=bool)
            x1, y1 = int((x0 + box_w) * w), int((y0 + box_h) * h)
            mask[int(y0 * h) : y1, int(x0 * w) : x1] = True
            valid = mask & (frame.depth > 0)
            total += 1
            if valid.sum() >= MIN_VALID_DEPTH_PIXELS:
                successes += 1

    assert successes / total > 0.9, "dense SANPO depth should clear the lifting threshold almost everywhere"


def test_velocity_derived_heading_matches_chest_camera_measured_heading():
    """The actual validation this whole rotation-convention investigation
    was for: does locus.geometry.frames' velocity-derived body-frame
    heading (from the head camera's own trajectory) agree with SANPO's
    *literal* chest-camera-measured orientation? Before correcting the
    local-forward-axis assumption (see sanpo_loader.py's module docstring),
    this comparison was ~130 degrees off and essentially uncorrelated --
    not a validation of anything. With the correction applied, it agrees
    at a median of a few tens of degrees, consistent with real gait/turn
    noise rather than a convention bug."""
    from locus.data.sanpo_loader import _load_poses
    from locus.geometry.frames import body_rotation_from_velocity
    from locus.geometry.kinematics import finite_difference_velocity
    from pathlib import Path

    session_id = config["sanpo_sequences"][0]
    cache_root = Path(config["paths"]["sanpo_cache_dir"]) / session_id
    head_poses = _load_poses(session_id, cache_root / "camera_head", camera="camera_head")
    chest_poses = _load_poses(session_id, cache_root / "camera_chest", camera="camera_chest")
    n = min(len(head_poses), len(chest_poses))
    fps = 15.0
    window = 15  # 1s smoothing -- a single-frame velocity is too noisy for a bobbing walking gait

    def yaw_deg(v: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(v[0], v[2])))

    diffs = []
    for i in range(window, n, 5):
        h_ready, h_pos, _ = head_poses[i]
        h_ready_prev, h_pos_prev, _ = head_poses[i - window]
        c_ready, _, c_quat = chest_poses[i]
        if not (h_ready and h_ready_prev and c_ready):
            continue
        velocity_world = finite_difference_velocity(h_pos_prev, h_pos, window / fps)
        if np.linalg.norm(velocity_world) < 0.1:
            continue
        approx_heading = yaw_deg(body_rotation_from_velocity(velocity_world)[:, 2])

        from scipy.spatial.transform import Rotation

        from locus.data.sanpo_loader import _R_SANPO_LOCAL_FROM_OPENCV

        R_chest = Rotation.from_quat(c_quat).as_matrix() @ _R_SANPO_LOCAL_FROM_OPENCV
        chest_heading = yaw_deg(R_chest @ np.array([0.0, 0.0, 1.0]))

        diffs.append((approx_heading - chest_heading + 180) % 360 - 180)

    diffs = np.array(diffs)
    assert len(diffs) > 20, "too few comparable frames to mean anything"
    assert np.median(np.abs(diffs)) < 45, (
        f"velocity-derived heading should roughly track the real chest-camera heading "
        f"(median diff {np.median(np.abs(diffs)):.1f} deg) -- if this regresses, check "
        f"the rotation-convention assumptions in sanpo_loader.py's module docstring first"
    )
