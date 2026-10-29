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
"""

from __future__ import annotations

import numpy as np

from locus.types import Pose

_WORLD_UP = np.array([0.0, 1.0, 0.0])


def ego_to_world(p_ego: np.ndarray, pose: Pose) -> np.ndarray:
    return pose.R_world_cam @ p_ego + pose.t_world_cam


def world_to_ego(p_world: np.ndarray, pose: Pose) -> np.ndarray:
    return pose.R_world_cam.T @ (p_world - pose.t_world_cam)


def body_rotation_from_velocity(
    velocity_world: np.ndarray, up_world: np.ndarray = _WORLD_UP
) -> np.ndarray:
    """Orthonormal world->body basis (as columns) with +Z = direction of travel."""
    speed = np.linalg.norm(velocity_world)
    forward = velocity_world / speed if speed > 1e-6 else np.array([0.0, 0.0, 1.0])
    right = np.cross(up_world, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1)  # R_world_body


def world_to_body(
    p_world: np.ndarray, body_origin_world: np.ndarray, velocity_world: np.ndarray
) -> np.ndarray:
    r_world_body = body_rotation_from_velocity(velocity_world)
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
