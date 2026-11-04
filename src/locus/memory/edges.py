"""Geometric edges from the wearer to every nearby node -- never learned."""

from __future__ import annotations

import numpy as np

from locus.geometry.frames import WORLD_DOWN_DEFAULT, bearing_rad, horizontal, world_to_body
from locus.geometry.kinematics import closing_speed, time_to_collision
from locus.memory.motion_classes import MOVABLE, motion_class
from locus.types import Edge, Node

WEARER_NODE_ID = 0

# A movable node unobserved for longer than this gets no edge at all: its
# remembered position plus a frozen velocity is a guess that gets worse
# every frame. It stays in the graph (and fades from the 3D view) until
# SpatialMemoryGraph evicts it.
MOVER_STALE_S = 0.75


def closest_approach(relative: np.ndarray, relative_velocity: np.ndarray) -> tuple[float, float | None]:
    """Miss distance and time to it for an object at `relative` (from the
    wearer) moving at `relative_velocity` (object minus wearer). If the two
    aren't getting any closer, the nearest they'll be is now: (range, None)."""
    speed_sq = float(relative_velocity @ relative_velocity)
    t = -float(relative @ relative_velocity) / speed_sq if speed_sq > 1e-9 else 0.0
    if t <= 0:
        return float(np.linalg.norm(relative)), None
    return float(np.linalg.norm(relative + relative_velocity * t)), t


def compute_edges(
    nodes: list[Node],
    wearer_position_world: np.ndarray,
    wearer_velocity_world: np.ndarray,
    edge_radius_m: float,
    world_down: np.ndarray = WORLD_DOWN_DEFAULT,
    wearer_heading_world: np.ndarray | None = None,
    t: float | None = None,
) -> list[Edge]:
    """`wearer_heading_world` orients the body frame (clock positions) and
    `wearer_velocity_world` drives closing speed; they differ when the
    wearer has stopped (see locus.geometry.kinematics.WearerMotion).
    Without a heading, the velocity's direction is used, as before.
    Without `t`, stale-mover exclusion is skipped."""
    heading = wearer_heading_world if wearer_heading_world is not None else wearer_velocity_world
    edges = []
    for node in nodes:
        if t is not None and motion_class(node.concept) == MOVABLE and t - node.last_seen > MOVER_STALE_S:
            continue
        relative = node.p_world - wearer_position_world
        range_m = float(np.linalg.norm(relative))
        if range_m > edge_radius_m:
            continue

        p_body = world_to_body(node.p_world, wearer_position_world, heading, world_down)
        cs = closing_speed(
            wearer_position_world, wearer_velocity_world, node.p_world, node.velocity_world
        )
        cpa_m, tcpa_s = closest_approach(
            horizontal(relative, world_down), horizontal(node.velocity_world - wearer_velocity_world, world_down)
        )
        edges.append(
            Edge(
                src_node_id=WEARER_NODE_ID,
                dst_node_id=node.node_id,
                range_m=range_m,
                bearing_rad=bearing_rad(p_body),
                elevation_delta_m=-float(np.dot(relative, world_down)),  # + = above the camera
                ttc_s=time_to_collision(range_m, cs),
                cpa_m=cpa_m,
                tcpa_s=tcpa_s,
            )
        )
    return edges
