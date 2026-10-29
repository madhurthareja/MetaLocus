"""Velocity by finite difference, closing speed along the line of sight,
and time-to-collision.

Closing speed is clamped at zero: a receding object has infinite TTC,
not negative -- negative TTC would read as "already collided", which is
wrong and would trigger spurious hazard alerts.
"""

from __future__ import annotations

import numpy as np


def finite_difference_velocity(
    p_prev: np.ndarray, p_curr: np.ndarray, dt: float
) -> np.ndarray:
    if dt <= 0:
        return np.zeros_like(p_curr)
    return (p_curr - p_prev) / dt


def closing_speed(
    wearer_p: np.ndarray,
    wearer_v: np.ndarray,
    object_p: np.ndarray,
    object_v: np.ndarray,
) -> float:
    """Rate at which range is shrinking, along the current line of sight."""
    relative_p = object_p - wearer_p
    range_m = np.linalg.norm(relative_p)
    if range_m < 1e-9:
        return 0.0
    line_of_sight = relative_p / range_m
    relative_v = object_v - wearer_v
    return -float(np.dot(relative_v, line_of_sight))


def time_to_collision(range_m: float, closing_speed_mps: float) -> float | None:
    """Seconds to collision, or None if not closing (closing_speed_mps <= 0)."""
    if closing_speed_mps <= 0:
        return None
    return range_m / closing_speed_mps
