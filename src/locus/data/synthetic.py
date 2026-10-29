"""Deterministic synthetic egocentric sequence generator.

No rendering engine -- built analytically so every ground-truth quantity
(visibility, world position, body-frame bearing, TTC) is exact, not
estimated. This is what tests/test_memory.py is tested against; it is
the only data source that doesn't need weights, keys or network access.

Camera yaw (head orientation) and the wearer's direction of travel are
modelled independently on purpose: the wearer always walks along world
+Z (velocity_world is constant), while the head can yaw away from that
direction. That's what makes `head_turn_away` / `head_turn_return`
meaningful tests of locus.geometry.frames' body-vs-ego distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from locus.geometry.frames import bearing_clockface, world_to_body, world_to_ego
from locus.geometry.intrinsics import Intrinsics
from locus.types import Pose

FPS = 30
DT = 1.0 / FPS
WALKING_SPEED_MPS = 1.4
FAR_PLANE_DEPTH = 50.0

DEFAULT_INTRINSICS = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

_CONCEPT_COLORS = {
    "chair": np.array([200, 80, 80]),
    "bicycle": np.array([80, 160, 200]),
    "table": np.array([120, 180, 90]),
}


@dataclass
class SyntheticObject:
    concept: str
    object_id: str
    position_world_t0: np.ndarray
    radius_m: float = 0.3
    velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def position_world(self, t: float) -> np.ndarray:
        return self.position_world_t0 + self.velocity_world * t


@dataclass
class GroundTruthFrame:
    frame_index: int
    t: float
    pose: Pose
    object_position_world: dict[str, np.ndarray]
    object_visible: dict[str, bool]
    object_bearing_body: dict[str, float]


@dataclass
class GroundTruth:
    objects: list[SyntheticObject]
    frames: list[GroundTruthFrame]
    events: dict[str, int]  # e.g. {"obj1_exit_frame": 45, "obj1_reentry_frame": 165}


def _rotation_about_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _render_frame(
    t: float,
    frame_index: int,
    camera_pos: np.ndarray,
    yaw_rad: float,
    objects: list[SyntheticObject],
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray, Pose, dict[str, bool], dict[str, tuple]]:
    """Returns (rgb, depth, pose, visibility, {object_id: (mask, box_xywh_norm, p_cam)})."""
    pose = Pose(timestamp=t, R_world_cam=_rotation_about_y(yaw_rad), t_world_cam=camera_pos)

    h, w = intrinsics.height, intrinsics.width
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.full((h, w), FAR_PLANE_DEPTH, dtype=float)
    visibility: dict[str, bool] = {}
    detections: dict[str, tuple] = {}

    # Paint farthest-first so a nearer object's disk correctly occludes a farther one.
    by_depth = []
    for obj in objects:
        p_world = obj.position_world(t)
        p_cam = world_to_ego(p_world, pose)
        by_depth.append((p_cam[2], obj, p_cam))
    by_depth.sort(key=lambda item: item[0], reverse=True)

    for depth_val, obj, p_cam in by_depth:
        if depth_val <= 0:
            visibility[obj.object_id] = False
            continue
        pixel = intrinsics.project(p_cam)
        pixel_radius = max(1.0, intrinsics.fx * obj.radius_m / depth_val)
        u, v = pixel

        visible = (
            intrinsics.in_frustum(pixel)
            and (u + pixel_radius) >= 0
            and (u - pixel_radius) <= w
            and (v + pixel_radius) >= 0
            and (v - pixel_radius) <= h
        )
        visibility[obj.object_id] = bool(visible)
        if not visible:
            continue

        yy, xx = np.ogrid[:h, :w]
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= pixel_radius**2
        color = _CONCEPT_COLORS.get(obj.concept, np.array([180, 180, 180]))
        rgb[disk] = color
        depth[disk] = depth_val

        x0 = max(0.0, (u - pixel_radius) / w)
        y0 = max(0.0, (v - pixel_radius) / h)
        x1 = min(1.0, (u + pixel_radius) / w)
        y1 = min(1.0, (v + pixel_radius) / h)
        box_xywh_norm = (x0, y0, x1 - x0, y1 - y0)
        detections[obj.object_id] = (disk, box_xywh_norm, p_cam)

    return rgb, depth, pose, visibility, detections


def _generate(
    objects: list[SyntheticObject],
    duration_s: float,
    yaw_fn,
    intrinsics: Intrinsics = DEFAULT_INTRINSICS,
    walking_speed_mps: float = WALKING_SPEED_MPS,
):
    from locus.types import Frame  # local import to avoid a cycle with Frame/Pose usage above

    n_frames = int(duration_s * FPS)
    frames: list[Frame] = []
    gt_frames: list[GroundTruthFrame] = []
    last_visible: dict[str, bool] = {obj.object_id: False for obj in objects}
    events: dict[str, int] = {}

    for i in range(n_frames):
        t = i * DT
        camera_pos = np.array([0.0, 0.0, walking_speed_mps * t])
        velocity_world = np.array([0.0, 0.0, walking_speed_mps])
        yaw_rad = yaw_fn(t)

        rgb, depth, pose, visibility, detections = _render_frame(
            t, i, camera_pos, yaw_rad, objects, intrinsics
        )

        object_position_world = {}
        object_bearing_body = {}
        for obj in objects:
            p_world = obj.position_world(t)
            object_position_world[obj.object_id] = p_world
            p_body = world_to_body(p_world, camera_pos, velocity_world)
            object_bearing_body[obj.object_id] = bearing_clockface(p_body)

            was_visible = last_visible[obj.object_id]
            now_visible = visibility[obj.object_id]
            if was_visible and not now_visible:
                events.setdefault(f"{obj.object_id}_exit_frame", i)
            if not was_visible and now_visible and f"{obj.object_id}_exit_frame" in events:
                events.setdefault(f"{obj.object_id}_reentry_frame", i)
            last_visible[obj.object_id] = now_visible

        frames.append(Frame(timestamp=t, rgb=rgb, depth=depth, pose=pose))
        gt_frames.append(
            GroundTruthFrame(
                frame_index=i,
                t=t,
                pose=pose,
                object_position_world=object_position_world,
                object_visible=dict(visibility),
                object_bearing_body=object_bearing_body,
            )
        )

    return frames, GroundTruth(objects=objects, frames=gt_frames, events=events)


def straight_walk(duration_s: float = 3.0, seed: int = 0):
    """Control case: no head turn, object stays in view throughout."""
    objects = [SyntheticObject("chair", "chair_1", np.array([2.0, 0.0, 8.0]))]
    return _generate(objects, duration_s, yaw_fn=lambda t: 0.0)


def head_turn_away(yaw_deg: float = 60.0, hold_s: float = 4.0, seed: int = 0):
    """The headline scenario: an object exits view as the head yaws away,
    while the wearer keeps walking straight ahead."""
    objects = [SyntheticObject("chair", "chair_1", np.array([2.0, 0.0, 8.0]))]
    lead_in_s, ramp_s = 1.0, 0.5

    def yaw_fn(t: float) -> float:
        if t < lead_in_s:
            frac = 0.0
        elif t < lead_in_s + ramp_s:
            frac = (t - lead_in_s) / ramp_s
        else:
            frac = 1.0
        return np.radians(yaw_deg * frac)

    duration_s = lead_in_s + ramp_s + hold_s
    return _generate(objects, duration_s, yaw_fn)


def head_turn_return(yaw_deg: float = 60.0, away_s: float = 4.0, seed: int = 0):
    """head_turn_away, then the head yaws back -- the re-identification test:
    the object re-entering view must bind to the same node_id."""
    objects = [SyntheticObject("chair", "chair_1", np.array([2.0, 0.0, 8.0]))]
    lead_in_s, ramp_s, settle_s = 1.0, 0.5, 2.0

    def yaw_fn(t: float) -> float:
        away_start = lead_in_s
        away_hold_start = lead_in_s + ramp_s
        return_start = away_hold_start + away_s
        return_end = return_start + ramp_s

        if t < away_start:
            frac = 0.0
        elif t < away_hold_start:
            frac = (t - away_start) / ramp_s
        elif t < return_start:
            frac = 1.0
        elif t < return_end:
            frac = 1.0 - (t - return_start) / ramp_s
        else:
            frac = 0.0
        return np.radians(yaw_deg * frac)

    duration_s = lead_in_s + ramp_s + away_s + ramp_s + settle_s
    return _generate(objects, duration_s, yaw_fn)


def cluttered_occlusion(duration_s: float = 2.0, seed: int = 0):
    """Two same-concept objects at different depths, overlapping in image
    space -- the scenario where mask-lifting should beat box-lifting."""
    objects = [
        SyntheticObject("chair", "chair_near", np.array([0.3, 0.0, 4.0]), radius_m=0.3),
        SyntheticObject("chair", "chair_far", np.array([0.5, 0.0, 7.0]), radius_m=0.3),
    ]
    return _generate(objects, duration_s, yaw_fn=lambda t: 0.0)


def crossing_hazard(speed_mps: float = 1.0, duration_s: float = 3.0, seed: int = 0):
    """A moving object on a collision course, for TTC tests. The wearer keeps
    walking forward at the usual pace while the object closes from ahead."""
    objects = [
        SyntheticObject(
            "bicycle",
            "bicycle_1",
            np.array([0.0, 0.0, 15.0]),
            radius_m=0.4,
            velocity_world=np.array([0.0, 0.0, -speed_mps]),
        )
    ]
    return _generate(objects, duration_s, yaw_fn=lambda t: 0.0)
