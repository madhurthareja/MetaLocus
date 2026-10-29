"""The A/B/C/D ablation ladder.

A: stateless per-frame captioner -- forgets everything every frame.
B: short sliding window -- survives a brief glance away, still forgets
   for good once the window elapses.
C: box-lifted instances through the persistent SpatialMemoryGraph.
D: the full system -- mask-lifted instances through the same graph.

C and D share one base class and differ only in which lifting function
they pass in, so the C-vs-D comparison isolates exactly one variable
(mask vs box) with everything else -- association, decay, eviction --
held identical. A and B share their own per-frame lift-and-locate logic
with C/D (`_instance_from_detection`) for the same reason: the only
difference between conditions should be the thing the ablation is
actually testing (persistence depth), not incidental implementation
drift between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from locus.geometry.frames import bearing_clockface, ego_to_world, world_to_body
from locus.geometry.intrinsics import Intrinsics
from locus.geometry.kinematics import finite_difference_velocity
from locus.geometry.lifting import lift, lift_from_box
from locus.memory.graph import SpatialMemoryGraph
from locus.types import Detection, Frame, Instance3D, Pose


@dataclass(frozen=True)
class ConditionAnswer:
    known: bool
    visible: bool
    bearing_clock: float | None
    range_m: float | None
    t_last_seen: float | None


def _instance_from_detection(
    detection: Detection,
    frame: Frame,
    lift_fn: Callable[[Detection, np.ndarray, Intrinsics], Instance3D | None],
    intrinsics: Intrinsics,
) -> Instance3D | None:
    if frame.depth is None or frame.pose is None:
        return None
    inst_cam = lift_fn(detection, frame.depth, intrinsics)
    if inst_cam is None:
        return None
    p_world = ego_to_world(inst_cam.p_cam, frame.pose)
    return Instance3D(
        concept=inst_cam.concept,
        instance_id=inst_cam.instance_id,
        p_cam=inst_cam.p_cam,
        p_world=p_world,
        extent=inst_cam.extent,
        score=inst_cam.score,
    )


class _PoseVelocityTracker:
    """Shared by every condition: wearer velocity_world via finite
    difference between consecutive poses, needed for body-frame bearing."""

    def __init__(self):
        self.prev_pose: Pose | None = None
        self.velocity_world = np.zeros(3)

    def reset(self):
        self.prev_pose = None
        self.velocity_world = np.zeros(3)

    def update(self, pose: Pose | None) -> None:
        if self.prev_pose is not None and pose is not None:
            dt = pose.timestamp - self.prev_pose.timestamp
            if dt > 0:
                self.velocity_world = finite_difference_velocity(
                    self.prev_pose.t_world_cam, pose.t_world_cam, dt
                )
        self.prev_pose = pose


_UNKNOWN = ConditionAnswer(known=False, visible=False, bearing_clock=None, range_m=None, t_last_seen=None)


class ConditionA:
    """Stateless per-frame captioner baseline: zero memory. The instant an
    object leaves the frame, this condition knows nothing about it -- that
    gap is exactly what persistence (C, D) is meant to fix."""

    def __init__(self, intrinsics: Intrinsics, lift_fn=lift):
        self._intrinsics = intrinsics
        self._lift_fn = lift_fn
        self._tracker = _PoseVelocityTracker()
        self._current: dict[str, Instance3D] = {}

    def reset(self) -> None:
        self._tracker.reset()
        self._current.clear()

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self._tracker.update(frame.pose)
        self._current.clear()  # the whole point: nothing survives past this frame
        for det in detections:
            inst = _instance_from_detection(det, frame, self._lift_fn, self._intrinsics)
            if inst is not None:
                self._current[det.concept] = inst

    def answer(self, concept: str) -> ConditionAnswer:
        inst = self._current.get(concept)
        if inst is None or self._tracker.prev_pose is None:
            return _UNKNOWN
        p_body = world_to_body(inst.p_world, self._tracker.prev_pose.t_world_cam, self._tracker.velocity_world)
        return ConditionAnswer(
            known=True,
            visible=True,
            bearing_clock=bearing_clockface(p_body),
            range_m=float(np.linalg.norm(inst.p_world - self._tracker.prev_pose.t_world_cam)),
            t_last_seen=self._tracker.prev_pose.timestamp,
        )


class ConditionB:
    """Short sliding-window baseline: keeps the last `window_s` seconds of
    detections, so it survives a brief glance away -- but unlike C/D it
    forgets for good once the window elapses. No notion of "it's still out
    there somewhere", just a window of recency."""

    def __init__(self, intrinsics: Intrinsics, window_s: float = 1.0, lift_fn=lift):
        self._intrinsics = intrinsics
        self._window_s = window_s
        self._lift_fn = lift_fn
        self._tracker = _PoseVelocityTracker()
        self._last_seen: dict[str, tuple[float, Instance3D]] = {}
        self._current_t = 0.0

    def reset(self) -> None:
        self._tracker.reset()
        self._last_seen.clear()
        self._current_t = 0.0

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self._tracker.update(frame.pose)
        self._current_t = frame.timestamp
        for det in detections:
            inst = _instance_from_detection(det, frame, self._lift_fn, self._intrinsics)
            if inst is not None:
                self._last_seen[det.concept] = (frame.timestamp, inst)

    def answer(self, concept: str) -> ConditionAnswer:
        entry = self._last_seen.get(concept)
        if entry is None or self._tracker.prev_pose is None:
            return _UNKNOWN
        obs_t, inst = entry
        elapsed = self._current_t - obs_t
        if elapsed > self._window_s:
            return _UNKNOWN
        p_body = world_to_body(inst.p_world, self._tracker.prev_pose.t_world_cam, self._tracker.velocity_world)
        return ConditionAnswer(
            known=True,
            visible=(elapsed == 0.0),
            bearing_clock=bearing_clockface(p_body),
            range_m=float(np.linalg.norm(inst.p_world - self._tracker.prev_pose.t_world_cam)),
            t_last_seen=obs_t,
        )


class _GraphCondition:
    """Shared base for C and D: identical persistent-memory pipeline. The
    only difference between the two subclasses is which lift function
    turns a Detection into an Instance3D."""

    def __init__(self, intrinsics: Intrinsics, lift_fn, **graph_kwargs):
        self._intrinsics = intrinsics
        self._lift_fn = lift_fn
        self._graph_kwargs = graph_kwargs
        self._graph = SpatialMemoryGraph(**graph_kwargs)
        self._tracker = _PoseVelocityTracker()

    def reset(self) -> None:
        self._graph = SpatialMemoryGraph(**self._graph_kwargs)
        self._tracker.reset()

    def observe(self, frame: Frame, detections: list[Detection]) -> None:
        self._tracker.update(frame.pose)
        instances = []
        for det in detections:
            inst = _instance_from_detection(det, frame, self._lift_fn, self._intrinsics)
            if inst is not None:
                instances.append(inst)
        self._graph.observe(instances, frame.pose, frame.timestamp)
        self._graph.step(frame.timestamp)

    def answer(self, concept: str) -> ConditionAnswer:
        nodes = self._graph.query_nodes(concept=concept)
        if not nodes or self._tracker.prev_pose is None:
            return _UNKNOWN
        node = max(nodes, key=lambda n: n.confidence)
        p_body = world_to_body(node.p_world, self._tracker.prev_pose.t_world_cam, self._tracker.velocity_world)
        return ConditionAnswer(
            known=True,
            visible=node.visible,
            bearing_clock=bearing_clockface(p_body),
            range_m=float(np.linalg.norm(node.p_world - self._tracker.prev_pose.t_world_cam)),
            t_last_seen=node.last_seen,
        )


class ConditionC(_GraphCondition):
    """Box-lifted instances through the persistent graph -- masks discarded."""

    def __init__(self, intrinsics: Intrinsics, **graph_kwargs):
        super().__init__(intrinsics, lift_fn=lift_from_box, **graph_kwargs)


class ConditionD(_GraphCondition):
    """The full system: mask-lifted instances through the persistent graph."""

    def __init__(self, intrinsics: Intrinsics, **graph_kwargs):
        super().__init__(intrinsics, lift_fn=lift, **graph_kwargs)
