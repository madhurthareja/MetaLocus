"""The spatial memory graph: the actual contribution of this project.

Persistence is the whole point (section 7.3 of the spec this was built
against): when an instance stops being observed, the node is NOT
deleted. Its last known world position is kept (correct forever for a
static object -- that's exactly why the world frame, not the camera
frame, is where state lives), its confidence decays by half-life, and it
is only evicted after a configured timeout with no observation.
"""

from __future__ import annotations

import numpy as np

from locus.geometry.frames import world_to_ego
from locus.geometry.intrinsics import Intrinsics
from locus.memory.decay import decayed_confidence
from locus.memory.edges import compute_edges
from locus.memory.node import NodeIdAllocator
from locus.types import Edge, Instance3D, Node, Pose

ASSOCIATION_GATE_M = 1.0


class SpatialMemoryGraph:
    def __init__(
        self,
        node_decay_half_life_seconds: float = 8.0,
        node_eviction_seconds: float = 20.0,
        edge_radius_m: float = 4.0,
        association_gate_m: float = ASSOCIATION_GATE_M,
    ):
        self.node_decay_half_life_seconds = node_decay_half_life_seconds
        self.node_eviction_seconds = node_eviction_seconds
        self.edge_radius_m = edge_radius_m
        self.association_gate_m = association_gate_m

        self.nodes: dict[int, Node] = {}
        self._instance_to_node: dict[int, int] = {}  # segmenter instance_id -> node_id
        self._allocator = NodeIdAllocator()

    def _find_spatial_match(self, instance: Instance3D) -> int | None:
        best_id, best_dist = None, self.association_gate_m
        for node_id, node in self.nodes.items():
            if node.concept != instance.concept:
                continue
            dist = float(np.linalg.norm(node.p_world - instance.p_world))
            if dist < best_dist:
                best_id, best_dist = node_id, dist
        return best_id

    def observe(self, instances: list[Instance3D], pose: Pose, t: float) -> None:
        for instance in instances:
            node_id = self._instance_to_node.get(instance.instance_id)
            if node_id is not None and node_id not in self.nodes:
                node_id = None  # evicted since; fall through to spatial/new path

            if node_id is None:
                node_id = self._find_spatial_match(instance)

            if node_id is None:
                node_id = self._allocator.allocate()
                self.nodes[node_id] = Node(
                    concept=instance.concept,
                    node_id=node_id,
                    p_world=instance.p_world.copy(),
                    first_seen=t,
                    last_seen=t,
                )
            else:
                node = self.nodes[node_id]
                dt = t - node.last_seen
                if dt > 0:
                    node.velocity_world = (instance.p_world - node.p_world) / dt
                node.p_world = instance.p_world.copy()
                node.last_seen = t
                node.observation_count += 1

            node = self.nodes[node_id]
            node.visible = True
            node.confidence = 1.0
            self._instance_to_node[instance.instance_id] = node_id

    def step(self, t: float) -> None:
        evicted = []
        for node_id, node in self.nodes.items():
            elapsed = t - node.last_seen
            if elapsed <= 0:
                continue  # observed at or after t -- nothing to decay yet
            node.visible = False
            node.confidence = decayed_confidence(elapsed, self.node_decay_half_life_seconds)
            if elapsed > self.node_eviction_seconds:
                evicted.append(node_id)

        for node_id in evicted:
            del self.nodes[node_id]
            stale_instances = [k for k, v in self._instance_to_node.items() if v == node_id]
            for k in stale_instances:
                del self._instance_to_node[k]

    def query_nodes(
        self,
        concept: str | None = None,
        visible: bool | None = None,
        max_range: float | None = None,
        wearer_position_world: np.ndarray | None = None,
    ) -> list[Node]:
        result = []
        for node in self.nodes.values():
            if concept is not None and node.concept != concept:
                continue
            if visible is not None and node.visible != visible:
                continue
            if max_range is not None:
                if wearer_position_world is None:
                    raise ValueError("max_range requires wearer_position_world")
                if np.linalg.norm(node.p_world - wearer_position_world) > max_range:
                    continue
            result.append(node)
        return result

    def edges(
        self, wearer_position_world: np.ndarray, wearer_velocity_world: np.ndarray
    ) -> list[Edge]:
        return compute_edges(
            list(self.nodes.values()),
            wearer_position_world,
            wearer_velocity_world,
            self.edge_radius_m,
        )

    def predict_reentry(
        self, t: float, pose: Pose, intrinsics: Intrinsics
    ) -> list[Node]:
        """Invisible nodes whose last known position would currently fall
        inside the frustum -- drives re-prompting the segmenter."""
        candidates = []
        for node in self.nodes.values():
            if node.visible:
                continue
            p_ego = world_to_ego(node.p_world, pose)
            if p_ego[2] <= 0:
                continue
            if intrinsics.in_frustum(intrinsics.project(p_ego)):
                candidates.append(node)
        return candidates

    def summary(self) -> dict:
        live = {n.concept for n in self.nodes.values() if n.visible}
        decaying = {n.concept for n in self.nodes.values() if not n.visible}
        return {
            "n_nodes": len(self.nodes),
            "concepts_live": sorted(live),
            "concepts_decaying": sorted(decaying - live),
        }
