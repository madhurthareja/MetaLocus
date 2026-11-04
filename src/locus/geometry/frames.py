"""Three reference frames, and the one invariant everything downstream
depends on.

- F_ego: camera/head frame, origin at the camera. What lifting.py produces.
- F_body: torso ("clock-face") frame. +Z is the direction of travel,
  derived from the smoothed world-frame velocity -- NOT from camera
  orientation. 12 o'clock is always "the way you're walking", regardless
  of which way the head is turned.
- F_world: the global frame the dataset's (or synthetic) odometry provides.

Invariant (tested in tests/test_geometry.py): for a static object and a
camera that only rotates (velocity_world unchanged), the body-frame
bearing stays constant while the ego-frame bearing changes with the
camera. If that test fails, nothing built on top of frames.py is
trustworthy.

Gravity is a per-dataset fact, not a constant, so every body-frame
function takes `world_down` (the world-frame unit vector pointing at the
ground). F_body itself is OpenCV-shaped -- +X right, +Y down, +Z
forward -- matching F_ego, so the same `bearing_rad` reads both.
- synthetic: its world IS the OpenCV camera frame at yaw 0, so world +Y
  points down -> (0, 1, 0), the default.
- SANPO and AEA: Z-up worlds -> (0, 0, -1). Using (0, 1, 0) there instead
  makes "right" a near-vertical axis built from velocity noise, so left
  and right stop meaning anything real (see docs/CLAUDE.md).
"""

from __future__ import annotations

import numpy as np

from locus.types import Pose

WORLD_DOWN_DEFAULT = np.array([0.0, 1.0, 0.0])  # synthetic / OpenCV-axis worlds


def ego_to_world(p_ego: np.ndarray, pose: Pose) -> np.ndarray:
    return pose.R_world_cam @ p_ego + pose.t_world_cam


def world_to_ego(p_world: np.ndarray, pose: Pose) -> np.ndarray:
    return pose.R_world_cam.T @ (p_world - pose.t_world_cam)


def horizontal(v: np.ndarray, world_down: np.ndarray = WORLD_DOWN_DEFAULT) -> np.ndarray:
    """`v` with its component along gravity removed."""
    return v - np.dot(v, world_down) * world_down


def body_rotation_from_velocity(
    velocity_world: np.ndarray, world_down: np.ndarray = WORLD_DOWN_DEFAULT
) -> np.ndarray:
    """Orthonormal world->body basis (as columns): +X right, +Y down, +Z =
    horizontal direction of travel. Only the direction of `velocity_world`
    is used, never its magnitude.

    No usable horizontal direction (standing still) falls back to an
    arbitrary horizontal axis so the basis stays finite -- callers that
    care (generate_dashboard_data.py) pass a held heading instead, see
    locus.geometry.kinematics.WearerMotion."""
    forward = horizontal(velocity_world, world_down)
    if np.linalg.norm(forward) < 1e-6:
        for candidate in np.eye(3)[::-1]:  # world Z, Y, X -- first one not parallel to gravity
            forward = horizontal(candidate, world_down)
            if np.linalg.norm(forward) > 1e-6:
                break
    forward = forward / np.linalg.norm(forward)
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)  # R_world_body


def world_to_body(
    p_world: np.ndarray,
    body_origin_world: np.ndarray,
    velocity_world: np.ndarray,
    world_down: np.ndarray = WORLD_DOWN_DEFAULT,
) -> np.ndarray:
    r_world_body = body_rotation_from_velocity(velocity_world, world_down)
    return r_world_body.T @ (p_world - body_origin_world)


def bearing_rad(p_body: np.ndarray) -> float:
    """Body-frame position -> signed angle from straight-ahead, radians."""
    x, _, z = p_body
    return float(np.arctan2(x, z))


def bearing_clockface(p_body: np.ndarray) -> float:
    """Body-frame position -> clock position in [1, 12]. 12 = straight ahead."""
    angle_deg = np.degrees(bearing_rad(p_body)) % 360.0
    clock = angle_deg / 30.0
    return clock if clock > 1e-9 else 12.0
