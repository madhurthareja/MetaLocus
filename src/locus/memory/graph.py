"""Persistent 3D spatial memory graph.

When an instance stops being observed, its node is not deleted: its last
known world position is kept (which is why state lives in the world
frame, not the camera frame), confidence decays by half-life, and the
node is evicted only after a configured timeout with no observation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from locus.geometry.frames import WORLD_DOWN_DEFAULT, world_to_ego
from locus.geometry.intrinsics import Intrinsics
from locus.memory.decay import decayed_confidence
from locus.memory.edges import compute_edges
from locus.memory.motion_classes import MOVABLE, STATIC, max_speed_mps, motion_class
from locus.memory.node import NodeIdAllocator
from locus.types import Edge, Instance3D, Node, Pose

ASSOCIATION_GATE_M = 1.0
VELOCITY_SMOOTHING_ALPHA = 0.3
# Movable things are only trustworthy while seen (see motion_classes.py):
# they fade and get evicted on a seconds scale, not the static default's.
MOVABLE_DECAY_HALF_LIFE_S = 1.0
MOVABLE_EVICTION_S = 4.0
# Mover velocity is a line fit over at most this much of the current track
# segment, reported only once the segment is long enough to mean anything.
MOVER_VELOCITY_WINDOW_S = 1.0
MOVER_VELOCITY_MIN_SPAN_S = 0.4
MOVER_VELOCITY_MIN_OBSERVATIONS = 5


class SpatialMemoryGraph:
    def __init__(
        self,
        node_decay_half_life_seconds: float = 8.0,
        node_eviction_seconds: float = 20.0,
        edge_radius_m: float = 4.0,
        association_gate_m: float = ASSOCIATION_GATE_M,
        world_down: np.ndarray = WORLD_DOWN_DEFAULT,
        movable_decay_half_life_seconds: float = MOVABLE_DECAY_HALF_LIFE_S,
        movable_eviction_seconds: float = MOVABLE_EVICTION_S,
    ):
        self.node_decay_half_life_seconds = node_decay_half_life_seconds
        self.node_eviction_seconds = node_eviction_seconds
        self.edge_radius_m = edge_radius_m
        self.association_gate_m = association_gate_m
        self.world_down = np.asarray(world_down, dtype=float)
        self.movable_decay_half_life_seconds = movable_decay_half_life_seconds
        self.movable_eviction_seconds = movable_eviction_seconds

        self.nodes: dict[int, Node] = {}
        self._instance_to_node: dict[int, int] = {}  # segmenter instance_id -> node_id
        self._allocator = NodeIdAllocator()
        self._mover_tracks: dict[int, deque[tuple[float, np.ndarray]]] = {}  # node_id -> (t, p_world)
        self._mover_velocity_ready: dict[int, bool] = {}

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
                kind = motion_class(node.concept)
                if kind == STATIC:
                    # Pinned, not smoothed: see motion_classes.py for the
                    # measured noise this replaces.
                    node.velocity_world = np.zeros(3)
                elif kind == MOVABLE:
                    if dt > 0:
                        self._update_mover_velocity(node, instance.p_world, t, dt)
                elif dt > 0:
                    # Exponential smoothing rather than a raw finite
                    # difference: mask/depth-lifting jitter would otherwise
                    # read as real velocity for an unclassified concept, and
                    # since a node's velocity is only updated here, one bad
                    # sample stays load-bearing for as long as the node is
                    # remembered.
                    instantaneous_v = (instance.p_world - node.p_world) / dt
                    node.velocity_world = (
                        VELOCITY_SMOOTHING_ALPHA * instantaneous_v
                        + (1 - VELOCITY_SMOOTHING_ALPHA) * node.velocity_world
                    )
                node.p_world = instance.p_world.copy()
                node.last_seen = t
                node.observation_count += 1

            node = self.nodes[node_id]
            node.visible = True
            node.confidence = 1.0
            self._instance_to_node[instance.instance_id] = node_id

        self._merge_duplicate_nodes()

    def _update_mover_velocity(self, node: Node, p_world: np.ndarray, t: float, dt: float) -> None:
        """Velocity for a movable node: the least-squares slope of its
        current consistent track segment, or zero until there is enough of
        one. A segmenter can hand one instance id to a different physical
        object (e.g. a tracker re-anchoring); a step faster than the class
        can move is treated as that kind of jump rather than motion and
        starts a new segment at zero velocity, so a stale/noisy estimate
        never carries over onto whatever the id lands on next. A line fit
        over the segment, instead of smoothed per-frame differences, keeps
        depth jitter from reading as speed."""
        track = self._mover_tracks.setdefault(node.node_id, deque([(node.last_seen, node.p_world.copy())]))
        if np.linalg.norm(p_world - node.p_world) / dt > max_speed_mps(node.concept):
            track.clear()
        track.append((t, p_world.copy()))
        while t - track[0][0] > MOVER_VELOCITY_WINDOW_S:
            track.popleft()

        ts = np.array([sample[0] for sample in track])
        ready = len(track) >= MOVER_VELOCITY_MIN_OBSERVATIONS and ts[-1] - ts[0] >= MOVER_VELOCITY_MIN_SPAN_S
        self._mover_velocity_ready[node.node_id] = ready
        if not ready:
            node.velocity_world = np.zeros(3)
            return
        ps = np.array([sample[1] for sample in track])
        tc = ts - ts.mean()
        node.velocity_world = (tc[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / (tc**2).sum()

    def velocity_measured(self, node_id: int) -> bool:
        """False while a movable node's zero velocity means "not measured
        yet" (track segment too short) rather than "not moving" -- the
        difference between "a bicycle" and "a stopped bicycle"."""
        if motion_class(self.nodes[node_id].concept) != MOVABLE:
            return True
        return self._mover_velocity_ready.get(node_id, False)

    def _merge_duplicate_nodes(self) -> None:
        """Two different instance_ids for the same physical object get
        their own node at creation time if they miss the association gate
        then (noisy depth, or a segmenter briefly emitting two overlapping
        detections for one object before its own de-duplication converges).
        The gate is only checked at creation, so such a pair can keep
        existing side by side even after later observations bring them
        within range of each other; this re-checks every same-concept pair
        after each observe() and merges any that now qualify."""
        by_concept: dict[str, list[int]] = {}
        for node_id, node in self.nodes.items():
            by_concept.setdefault(node.concept, []).append(node_id)

        merged: set[int] = set()
        for node_ids in by_concept.values():
            for i, id_a in enumerate(node_ids):
                if id_a in merged:
                    continue
                for id_b in node_ids[i + 1 :]:
                    if id_b in merged:
                        continue
                    dist = float(np.linalg.norm(self.nodes[id_a].p_world - self.nodes[id_b].p_world))
                    if dist < self.association_gate_m:
                        self._merge_into(id_a, id_b)
                        if id_a not in self.nodes:
                            # _merge_into kept id_b (it was observed more
                            # recently) and dropped id_a instead -- id_a is
                            # gone, so stop comparing against it.
                            merged.add(id_a)
                            break
                        merged.add(id_b)

    def _merge_into(self, keep: int, drop: int) -> None:
        keeper, loser = self.nodes[keep], self.nodes[drop]
        if loser.last_seen > keeper.last_seen:
            keep, drop = drop, keep
            keeper, loser = loser, keeper

        keeper.observation_count += loser.observation_count
        keeper.first_seen = min(keeper.first_seen, loser.first_seen)
        keeper.confidence = max(keeper.confidence, loser.confidence)

        for instance_id, node_id in self._instance_to_node.items():
            if node_id == drop:
                self._instance_to_node[instance_id] = keep
        del self.nodes[drop]
        # Two sources now feed one node; neither track segment describes it.
        for node_id in (keep, drop):
            self._mover_tracks.pop(node_id, None)
            self._mover_velocity_ready.pop(node_id, None)

    def step(self, t: float) -> None:
        evicted = []
        for node_id, node in self.nodes.items():
            elapsed = t - node.last_seen
            if elapsed <= 0:
                continue  # observed at or after t -- nothing to decay yet
            node.visible = False
            movable = motion_class(node.concept) == MOVABLE
            half_life = self.movable_decay_half_life_seconds if movable else self.node_decay_half_life_seconds
            eviction = self.movable_eviction_seconds if movable else self.node_eviction_seconds
            node.confidence = decayed_confidence(elapsed, half_life)
            if elapsed > eviction:
                evicted.append(node_id)

        for node_id in evicted:
            del self.nodes[node_id]
            self._mover_tracks.pop(node_id, None)
            self._mover_velocity_ready.pop(node_id, None)
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
        self,
        wearer_position_world: np.ndarray,
        wearer_velocity_world: np.ndarray,
        wearer_heading_world: np.ndarray | None = None,
        t: float | None = None,
    ) -> list[Edge]:
        return compute_edges(
            list(self.nodes.values()),
            wearer_position_world,
            wearer_velocity_world,
            self.edge_radius_m,
            world_down=self.world_down,
            wearer_heading_world=wearer_heading_world,
            t=t,
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
