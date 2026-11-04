import numpy as np
import pytest

from locus.geometry.frames import bearing_clockface, world_to_body, world_to_ego
from locus.geometry.intrinsics import Intrinsics
from locus.geometry.kinematics import closing_speed, time_to_collision
from locus.geometry.lifting import lift, lift_from_box
from locus.types import Detection, Pose


def _rotation_about_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_body_frame_bearing_constant_under_pure_camera_rotation():
    """The invariant everything else depends on: a static object's body-frame
    bearing must not move when only the camera rotates (walking direction
    unchanged); its ego-frame bearing must."""
    object_world = np.array([5.0, 0.0, 10.0])
    camera_pos = np.zeros(3)
    velocity_world = np.array([0.0, 0.0, 1.4])  # walking straight ahead, world +Z

    ego_bearings = []
    body_bearings = []
    for yaw_deg in [0, 15, 30, 45, -20]:
        R_world_cam = _rotation_about_y(np.radians(yaw_deg))
        pose = Pose(timestamp=0.0, R_world_cam=R_world_cam, t_world_cam=camera_pos)

        p_ego = world_to_ego(object_world, pose)
        ego_bearings.append(bearing_clockface(p_ego))

        p_body = world_to_body(object_world, camera_pos, velocity_world)
        body_bearings.append(bearing_clockface(p_body))

    assert len(set(round(b, 6) for b in body_bearings)) == 1, "body bearing must be constant"
    assert len(set(round(b, 6) for b in ego_bearings)) == len(ego_bearings), (
        "ego bearing must change with every distinct camera yaw"
    )


def test_bearing_clockface_cardinal_directions():
    assert bearing_clockface(np.array([0.0, 0.0, 1.0])) == pytest.approx(12.0)
    assert bearing_clockface(np.array([1.0, 0.0, 0.0])) == pytest.approx(3.0)
    assert bearing_clockface(np.array([0.0, 0.0, -1.0])) == pytest.approx(6.0)
    assert bearing_clockface(np.array([-1.0, 0.0, 0.0])) == pytest.approx(9.0)


def _make_depth_scene(width=64, height=64, fg_depth=2.0, bg_depth=10.0):
    depth = np.full((height, width), bg_depth, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    mask[20:40, 20:40] = True  # a 20x20 foreground square
    depth[mask] = fg_depth
    return depth, mask


def test_mask_lifting_more_accurate_than_box_lifting_on_occlusion():
    """Condition D (mask) must beat condition C (box) when the box's bounding
    rectangle includes background pixels the mask excludes -- this is the
    scenario the whole segmentation-grounded-memory claim rests on."""
    depth, fg_mask = _make_depth_scene()
    intrinsics = Intrinsics(fx=500, fy=500, cx=32, cy=32, width=64, height=64)

    # A box loosely bounding the foreground square, padded with background.
    box_xywh_norm = (15 / 64, 15 / 64, 30 / 64, 30 / 64)

    detection = Detection(
        concept="chair",
        instance_id=1,
        mask=fg_mask,
        box_xywh_norm=box_xywh_norm,
        score=0.9,
    )

    mask_instance = lift(detection, depth, intrinsics)
    box_instance = lift_from_box(detection, depth, intrinsics)

    true_depth = 2.0
    mask_error = abs(mask_instance.p_cam[2] - true_depth)
    box_error = abs(box_instance.p_cam[2] - true_depth)

    assert mask_error < box_error
    assert mask_error == pytest.approx(0.0, abs=1e-6)


def test_time_to_collision_matches_analytic_crossing_hazard():
    wearer_p = np.zeros(3)
    wearer_v = np.zeros(3)
    object_p = np.array([0.0, 0.0, 10.0])
    object_v = np.array([0.0, 0.0, -1.0])  # closing at 1 m/s

    cs = closing_speed(wearer_p, wearer_v, object_p, object_v)
    assert cs == pytest.approx(1.0)

    ttc = time_to_collision(range_m=10.0, closing_speed_mps=cs)
    assert ttc == pytest.approx(10.0)


def test_time_to_collision_is_none_when_receding():
    assert time_to_collision(range_m=10.0, closing_speed_mps=0.0) is None
    assert time_to_collision(range_m=10.0, closing_speed_mps=-1.0) is None


def test_body_frame_handedness_in_a_z_up_world():
    """SANPO and AEA are Z-up, unlike synthetic's OpenCV-axis (Y-down)
    world. Walking along world +Y in a right-handed Z-up world, world +X is
    on the wearer's right -- the same physical fact the synthetic world
    encodes as +X right of +Z travel. Passing the wrong gravity axis here
    is exactly what made SANPO's left/right a coin flip."""
    from locus.geometry.frames import bearing_rad

    z_down = np.array([0.0, 0.0, -1.0])
    walking = np.array([0.0, 1.4, 0.0])
    right_of_path = world_to_body(np.array([2.0, 5.0, 0.0]), np.zeros(3), walking, z_down)
    left_of_path = world_to_body(np.array([-2.0, 5.0, 0.0]), np.zeros(3), walking, z_down)
    overhead = world_to_body(np.array([0.0, 5.0, 2.0]), np.zeros(3), walking, z_down)

    assert bearing_rad(right_of_path) > 0
    assert bearing_rad(left_of_path) < 0
    due_right = world_to_body(np.array([3.0, 0.0, 0.0]), np.zeros(3), walking, z_down)
    assert bearing_clockface(due_right) == pytest.approx(3.0)
    assert bearing_rad(overhead) == pytest.approx(0.0, abs=1e-9), "height must not leak into left/right"
    assert overhead[1] < 0, "body +Y is down, so something overhead has negative body Y"


def test_body_frame_stays_finite_when_standing_still():
    z_down = np.array([0.0, 0.0, -1.0])
    p_body = world_to_body(np.array([1.0, 2.0, 0.0]), np.zeros(3), np.zeros(3), z_down)
    assert np.all(np.isfinite(p_body))


def test_wearer_motion_holds_heading_through_a_stop_and_ignores_repeated_poses():
    """Real SANPO pose streams repeat a pose on NOT_READY frames and stop
    walking for long stretches; neither may zero out the heading."""
    from locus.geometry.kinematics import WearerMotion

    motion = WearerMotion(world_down=np.array([0.0, 0.0, -1.0]))
    dt, pos = 1 / 15, np.zeros(3)
    for i in range(30):  # walk +Y at 1.2 m/s, every third pose a stale repeat
        if i % 3 != 2:
            pos = np.array([0.0, 1.2 * i * dt, 0.02 * np.sin(i)])  # small vertical gait bob
        velocity, heading = motion.update(i * dt, pos)
    assert np.linalg.norm(velocity) == pytest.approx(1.2, rel=0.15)
    assert heading == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=0.05)
    assert abs(velocity[2]) < 1e-9, "velocity is horizontal only"

    for i in range(30, 60):  # standing still
        velocity, heading = motion.update(i * dt, pos)
    assert np.linalg.norm(velocity) == 0.0
    assert heading == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=0.05)


def test_wearer_motion_uses_camera_forward_before_first_step():
    from locus.geometry.kinematics import WearerMotion

    motion = WearerMotion(world_down=np.array([0.0, 0.0, -1.0]))
    _, heading = motion.update(0.0, np.zeros(3), camera_forward_world=np.array([1.0, 0.0, -0.5]))
    assert heading == pytest.approx(np.array([1.0, 0.0, 0.0]))
