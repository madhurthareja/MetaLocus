"""Geometric edges from the wearer to every nearby node -- never learned."""

from __future__ import annotations

import numpy as np

from locus.geometry.frames import bearing_rad, world_to_body
from locus.geometry.kinematics import closing_speed, time_to_collision
from locus.types import Edge, Node

WEARER_NODE_ID = 0


def compute_edges(
    nodes: list[Node],
    wearer_position_world: np.ndarray,
    wearer_velocity_world: np.ndarray,
    edge_radius_m: float,
) -> list[Edge]:
    edges = []
    for node in nodes:
        relative = node.p_world - wearer_position_world
        range_m = float(np.linalg.norm(relative))
        if range_m > edge_radius_m:
            continue

        p_body = world_to_body(node.p_world, wearer_position_world, wearer_velocity_world)
        cs = closing_speed(
            wearer_position_world, wearer_velocity_world, node.p_world, node.velocity_world
        )
        edges.append(
            Edge(
                src_node_id=WEARER_NODE_ID,
                dst_node_id=node.node_id,
                range_m=range_m,
                bearing_rad=bearing_rad(p_body),
                elevation_delta_m=float(relative[1]),
                ttc_s=time_to_collision(range_m, cs),
            )
        )
    return edges
