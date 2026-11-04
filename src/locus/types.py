"""Core data types shared across all of Locus.

These are the contract between perception, geometry, memory and eval --
every real backend and every mock produces/consumes exactly these. Pure
dataclasses + numpy; no torch, no device, no vendor SDK import here, so
this module (and everything that only depends on it) stays usable on
either machine regardless of which one has a GPU.

Node is intentionally mutable (not frozen): the memory graph updates
confidence, position and visibility on the same object in place rather
than reconstructing nodes every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Detection:
    concept: str
    instance_id: int
    mask: np.ndarray  # bool, (H, W)
    box_xywh_norm: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class Pose:
    timestamp: float
    R_world_cam: np.ndarray  # (3, 3)
    t_world_cam: np.ndarray  # (3,)


@dataclass(frozen=True)
class Frame:
    timestamp: float
    rgb: np.ndarray  # (H, W, 3)
    depth: np.ndarray | None  # (H, W), metres
    pose: Pose | None


@dataclass(frozen=True)
class Instance3D:
    concept: str
    instance_id: int
    p_cam: np.ndarray  # (3,)
    p_world: np.ndarray  # (3,)
    extent: np.ndarray  # (3,)
    score: float


@dataclass
class Node:
    concept: str
    node_id: int
    p_world: np.ndarray  # (3,)
    velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    first_seen: float = 0.0
    last_seen: float = 0.0
    visible: bool = True
    confidence: float = 1.0
    observation_count: int = 1


@dataclass(frozen=True)
class Edge:
    src_node_id: int
    dst_node_id: int
    range_m: float
    bearing_rad: float
    elevation_delta_m: float
    ttc_s: float | None
    # Closest point of approach, horizontal, if both keep their current
    # velocities: how far apart the wearer and the object will be at their
    # nearest, and when. A closing object can still pass well clear (a pole
    # beside the path); see locus.actuation.directives.on_collision_course.
    cpa_m: float | None = None
    tcpa_s: float | None = None


@dataclass(frozen=True)
class Query:
    text: str
    timestamp: float
    frame_hint: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    node_ids: list[int]
    bearing_rad: float | None = None
    range_m: float | None = None
