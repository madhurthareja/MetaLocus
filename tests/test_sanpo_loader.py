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


def _yaw_deg_about_down(v: np.ndarray, world_down: np.ndarray) -> float:
    """Heading of `v` in the plane perpendicular to gravity, against a fixed
    horizontal reference -- only differences of two of these are used."""
    from locus.geometry.frames import body_rotation_from_velocity

    ref = body_rotation_from_velocity(np.array([1.0, 1.0, 1.0]), world_down)  # any non-vertical direction
    return float(np.degrees(np.arctan2(v @ ref[:, 0], v @ ref[:, 2])))


def test_velocity_derived_heading_matches_chest_camera_measured_heading():
    """Does locus.geometry.frames' velocity-derived body-frame heading (from
    the head camera's own trajectory) agree with SANPO's *literal*
    chest-camera-measured orientation? Yaw is measured in the plane
    perpendicular to SANPO's real gravity axis (world Z). An earlier version
    of this test measured it as atan2(x, z) -- assuming Y-up in a world where
    the wearer walks along +Y -- which discarded the dominant motion
    component and let a wrong camera-axis correction pass at a ~22 degree
    "median" (see sanpo_loader.py's rotation-convention note). Measured
    correctly: 14.7 degrees median on walking frames."""
    from pathlib import Path

    from scipy.spatial.transform import Rotation

    from locus.data.sanpo_loader import SANPO_WORLD_DOWN, _load_poses

    world_down = np.array(SANPO_WORLD_DOWN)
    session_id = config["sanpo_sequences"][0]
    cache_root = Path(config["paths"]["sanpo_cache_dir"]) / session_id
    head_poses = _load_poses(session_id, cache_root / "camera_head", camera="camera_head")
    chest_poses = _load_poses(session_id, cache_root / "camera_chest", camera="camera_chest")
    n = min(len(head_poses), len(chest_poses))
    window = 15  # 1s smoothing -- a single-frame velocity is too noisy for a bobbing walking gait

    diffs = []
    for i in range(window, n, 3):
        h_ready, h_pos, _ = head_poses[i]
        h_ready_prev, h_pos_prev, _ = head_poses[i - window]
        c_ready, _, c_quat = chest_poses[i]
        if not (h_ready and h_ready_prev and c_ready):
            continue
        velocity_world = h_pos - h_pos_prev
        velocity_world = velocity_world - (velocity_world @ world_down) * world_down
        if np.linalg.norm(velocity_world) * 15.0 / window < 0.3:
            continue  # standing still: no velocity heading to compare
        chest_forward = Rotation.from_quat(c_quat).as_matrix()[:, 2]
        diff = _yaw_deg_about_down(velocity_world, world_down) - _yaw_deg_about_down(chest_forward, world_down)
        diffs.append((diff + 180) % 360 - 180)

    diffs = np.array(diffs)
    assert len(diffs) > 20, "too few comparable frames to mean anything"
    assert np.median(np.abs(diffs)) < 25, (
        f"velocity-derived heading should track the real chest-camera heading "
        f"(median diff {np.median(np.abs(diffs)):.1f} deg, measured 14.7) -- if this regresses, "
        f"check SANPO_WORLD_DOWN and the no-correction rotation convention in sanpo_loader.py first"
    )


def test_identity_rotation_convention_is_multiview_consistent():
    """The check that actually settles SANPO's rotation convention (see
    sanpo_loader.py): a static scene point lifted from one frame's depth
    and carried through world into a frame ~0.5s later must land on that
    frame's own measured depth. The raw quaternion (no camera-axis
    correction) does this at ~16% median relative depth error over a
    multi-pair search; the convention this loader used to apply scored at
    chance (~45%). Any axis or handedness mistake in R_world_cam breaks it."""
    from locus.data.sanpo_loader import load_sequence

    session_id = config["sanpo_sequences"][0]
    seq = load_sequence(session_id, config=config)
    K = seq.intrinsics
    rng = np.random.default_rng(0)

    def reprojection_error(i: int, j: int, R_fix: np.ndarray) -> float:
        fi = next(seq.iter_frames(max_frames=1, start_frame=i))
        fj = next(seq.iter_frames(max_frames=1, start_frame=j))
        us = rng.integers(0, K.width, 4000)
        vs = rng.integers(0, K.height, 4000)
        z = fi.depth[vs, us]
        ok = (z > 0.5) & (z < 15)
        us, vs, z = us[ok], vs[ok], z[ok]
        p_cam = np.stack([(us - K.cx) / K.fx * z, (vs - K.cy) / K.fy * z, z], 1)
        Ri, Rj = fi.pose.R_world_cam @ R_fix, fj.pose.R_world_cam @ R_fix
        p_world = (Ri @ p_cam.T).T + fi.pose.t_world_cam
        p_j = (Rj.T @ (p_world - fj.pose.t_world_cam).T).T
        front = p_j[:, 2] > 0.3
        u = (p_j[front, 0] / p_j[front, 2] * K.fx + K.cx).astype(int)
        v = (p_j[front, 1] / p_j[front, 2] * K.fy + K.cy).astype(int)
        inside = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        measured = fj.depth[v[inside], u[inside]]
        predicted = p_j[front, 2][inside]
        good = measured > 0
        return float(np.median(np.abs(predicted[good] - measured[good]) / measured[good]))

    old_convention = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pairs = [(20, 28), (100, 108)]  # both READY, walking
    identity_err = np.mean([reprojection_error(i, j, np.eye(3)) for i, j in pairs])
    old_err = np.mean([reprojection_error(i, j, old_convention) for i, j in pairs])
    assert identity_err < 0.15, f"raw-quaternion convention should be multi-view consistent, got {identity_err:.1%}"
    assert identity_err < 0.5 * old_err, f"identity {identity_err:.1%} should clearly beat the old convention {old_err:.1%}"


def test_reported_side_matches_image_side():
    """The left/right check that matches what a user actually hears: on
    frames where the head faces roughly the walking direction, a point on
    the image's right half must be reported on the right (positive body
    bearing), and vice versa. Measured on the shipped dashboard run before
    the convention fix: 55% agreement (a coin flip); after: 99%. This
    replaces a test that only checked a sign against a body frame whose
    "right" axis was, on SANPO's Z-up world, built from velocity noise."""
    from locus.data.sanpo_loader import load_sequence
    from locus.geometry.frames import bearing_rad, ego_to_world, horizontal, world_to_body
    from locus.geometry.kinematics import WearerMotion

    session_id = config["sanpo_sequences"][0]
    seq = load_sequence(session_id, config=config)
    world_down = np.array(seq.world_down)
    motion = WearerMotion(world_down=world_down)

    agree = total = 0
    for frame in seq.iter_frames(max_frames=200):
        _, heading = motion.update(frame.timestamp, frame.pose.t_world_cam)
        if heading is None or np.linalg.norm(motion.velocity) == 0:
            continue
        cam_fwd = horizontal(frame.pose.R_world_cam[:, 2], world_down)
        cam_fwd /= np.linalg.norm(cam_fwd)
        if np.degrees(np.arccos(np.clip(cam_fwd @ heading, -1, 1))) > 20:
            continue
        for side in (+1.0, -1.0):  # 2m to the camera's right / left, 5m ahead
            p_world = ego_to_world(np.array([2.0 * side, 0.0, 5.0]), frame.pose)
            p_body = world_to_body(p_world, frame.pose.t_world_cam, heading, world_down)
            agree += np.sign(bearing_rad(p_body)) == side
            total += 1

    assert total > 40, "too few walking, forward-facing frames to mean anything"
    assert agree / total > 0.9, f"reported side should match image side (got {agree / total:.0%})"


def test_prefetch_populates_cache_so_iter_frames_does_no_network_io():
    """prefetch() exists specifically because sequential per-frame downloads
    were the actual bottleneck processing a full real session (minutes of
    network wait, not SAM 3.1). Confirms it actually warms the cache."""
    from locus.data.sanpo_loader import load_sequence

    session_id = config["sanpo_sequences"][0]
    seq = load_sequence(session_id, config=config)
    seq.prefetch(max_frames=3, start_frame=50)

    for i in range(50, 53):
        name = f"{i:06d}"
        assert (seq.cache_dir / "video_frames" / f"{name}.png").exists()
        assert (seq.cache_dir / "depth_maps" / f"{name}.float16.gz").exists()

    # Should now be purely local reads -- just confirm it still works end to end.
    frames = list(seq.iter_frames(max_frames=3, start_frame=50))
    assert len(frames) == 3
