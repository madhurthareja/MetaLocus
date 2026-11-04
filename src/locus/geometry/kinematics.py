"""Velocity by finite difference, closing speed along the line of sight,
and time-to-collision.

Closing speed is clamped at zero: a receding object has infinite TTC,
not negative -- negative TTC would read as "already collided", which is
wrong and would trigger spurious hazard alerts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from locus.geometry.frames import WORLD_DOWN_DEFAULT, horizontal

WEARER_VELOCITY_WINDOW_S = 1.0
WEARER_MIN_WALKING_SPEED_MPS = 0.3


class WearerMotion:
    """Wearer velocity and heading from a stream of positions.

    A one-frame finite difference isn't usable on real pose data: a
    stationary wearer, or a loader substituting a not-ready pose row with
    the nearest ready one, both repeat a position between frames, reading
    as zero velocity and collapsing the body frame to an arbitrary world
    axis. This uses a causal window instead, and keeps two things apart
    that a naive velocity estimate conflates:
    - `velocity`: real horizontal velocity, zero when standing still --
      what closing speed / TTC needs.
    - `heading`: unit horizontal direction of travel, held through stops
      (a person who pauses still faces the way they were walking) -- what
      the body frame needs. Before the first walking sample it falls back
      to the camera's own horizontal forward, if given."""

    def __init__(
        self,
        world_down: np.ndarray = WORLD_DOWN_DEFAULT,
        window_s: float = WEARER_VELOCITY_WINDOW_S,
        min_speed_mps: float = WEARER_MIN_WALKING_SPEED_MPS,
    ):
        self.world_down = np.asarray(world_down, dtype=float)
        self.window_s = window_s
        self.min_speed_mps = min_speed_mps
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._held_heading: np.ndarray | None = None
        self.velocity = np.zeros(3)
        self.heading: np.ndarray | None = None

    def update(
        self, t: float, position_world: np.ndarray, camera_forward_world: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._history.append((t, np.asarray(position_world, dtype=float)))
        while len(self._history) > 2 and t - self._history[1][0] >= self.window_s:
            self._history.popleft()

        t0, p0 = self._history[0]
        v = horizontal(finite_difference_velocity(p0, self._history[-1][1], t - t0), self.world_down)
        speed = float(np.linalg.norm(v))
        if speed >= self.min_speed_mps:
            self.velocity = v
            self._held_heading = v / speed
        else:
            self.velocity = np.zeros(3)

        self.heading = self._held_heading
        if self.heading is None and camera_forward_world is not None:
            fwd = horizontal(np.asarray(camera_forward_world, dtype=float), self.world_down)
            if np.linalg.norm(fwd) > 1e-6:
                self.heading = fwd / np.linalg.norm(fwd)
        return self.velocity, self.heading


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
