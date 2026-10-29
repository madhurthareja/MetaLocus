import numpy as np
import pytest

from locus.data.synthetic import (
    cluttered_occlusion,
    crossing_hazard,
    head_turn_away,
    head_turn_return,
)
from locus.geometry.frames import bearing_clockface, world_to_body
from locus.geometry.intrinsics import Intrinsics
from locus.geometry.kinematics import closing_speed, time_to_collision
from locus.geometry.lifting import lift, lift_from_box
from locus.memory.graph import SpatialMemoryGraph
from locus.types import Detection, Instance3D

WALKING_SPEED = 1.4
_OBJECT_ID_TO_INT = {}


def _stable_instance_id(object_id: str) -> int:
    return _OBJECT_ID_TO_INT.setdefault(object_id, len(_OBJECT_ID_TO_INT) + 1)


def _instances_for_frame(gt_frame, objects) -> list[Instance3D]:
    """Stand-in for a MockSegmenter: perfect detections for every visible
    ground-truth object, with a stable instance_id per object across frames."""
    instances = []
    for obj in objects:
        if not gt_frame.object_visible.get(obj.object_id, False):
            continue
        instances.append(
            Instance3D(
                concept=obj.concept,
                instance_id=_stable_instance_id(obj.object_id),
                p_cam=np.zeros(3),
                p_world=gt_frame.object_position_world[obj.object_id],
                extent=np.zeros(3),
                score=1.0,
            )
        )
    return instances


def _run_scenario(graph: SpatialMemoryGraph, gt):
    for gt_frame in gt.frames:
        instances = _instances_for_frame(gt_frame, gt.objects)
        graph.observe(instances, gt_frame.pose, gt_frame.t)
        graph.step(gt_frame.t)
    return gt


def test_node_persists_after_head_turn_with_correct_world_position():
    _, gt = head_turn_away(yaw_deg=60.0, hold_s=4.0)
    graph = SpatialMemoryGraph(node_eviction_seconds=20.0)
    _run_scenario(graph, gt)

    assert "chair_1_exit_frame" in gt.events, "scenario didn't actually take the object out of view"

    nodes = graph.query_nodes(concept="chair")
    assert len(nodes) == 1
    node = nodes[0]
    assert node.visible is False

    true_position = gt.objects[0].position_world(gt.frames[-1].t)
    assert node.p_world == pytest.approx(true_position, abs=1e-6)


def test_body_frame_bearing_tracks_ground_truth_while_unseen():
    _, gt = head_turn_away(yaw_deg=60.0, hold_s=4.0)
    graph = SpatialMemoryGraph()
    _run_scenario(graph, gt)

    exit_frame = gt.events["chair_1_exit_frame"]
    node = graph.query_nodes(concept="chair")[0]

    # Spot-check a few frames after exit: the wearer keeps walking, so the
    # wearer's position changes, but the node's stored world position doesn't
    # (static object) -- recomputed body-frame bearing from it must still
    # match ground truth's independently-computed bearing at each of those
    # later times.
    for gt_frame in gt.frames[exit_frame + 1 :: 10]:
        wearer_pos = gt_frame.pose.t_world_cam
        velocity_world = np.array([0.0, 0.0, WALKING_SPEED])
        p_body = world_to_body(node.p_world, wearer_pos, velocity_world)
        recomputed_bearing = bearing_clockface(p_body)
        true_bearing = gt_frame.object_bearing_body["chair_1"]
        assert recomputed_bearing == pytest.approx(true_bearing, abs=1e-6)


def test_confidence_decays_monotonically_and_resets_on_reobservation():
    graph = SpatialMemoryGraph(node_decay_half_life_seconds=2.0, node_eviction_seconds=100.0)
    instance = Instance3D(
        concept="chair", instance_id=1, p_cam=np.zeros(3), p_world=np.array([1.0, 0.0, 5.0]),
        extent=np.zeros(3), score=1.0,
    )
    pose = None  # observe() doesn't dereference pose in this path
    graph.observe([instance], pose, t=0.0)
    node_id = next(iter(graph.nodes))

    confidences = []
    for t in [0.5, 1.0, 2.0, 4.0]:
        graph.step(t)
        confidences.append(graph.nodes[node_id].confidence)

    assert confidences == sorted(confidences, reverse=True)
    assert all(c < 1.0 for c in confidences)

    graph.observe([instance], pose, t=5.0)
    assert graph.nodes[node_id].confidence == 1.0


def test_node_evicted_after_timeout_not_before():
    graph = SpatialMemoryGraph(node_eviction_seconds=1.0)
    instance = Instance3D(
        concept="chair", instance_id=1, p_cam=np.zeros(3), p_world=np.array([1.0, 0.0, 5.0]),
        extent=np.zeros(3), score=1.0,
    )
    graph.observe([instance], None, t=0.0)
    node_id = next(iter(graph.nodes))

    graph.step(0.5)
    assert node_id in graph.nodes, "evicted too early"

    graph.step(1.5)
    assert node_id not in graph.nodes, "not evicted after the configured timeout"


def test_reentry_binds_to_same_node_id():
    global _OBJECT_ID_TO_INT
    _OBJECT_ID_TO_INT = {}
    _, gt = head_turn_return(yaw_deg=60.0, away_s=4.0)
    graph = SpatialMemoryGraph(node_eviction_seconds=20.0)
    _run_scenario(graph, gt)

    assert "chair_1_exit_frame" in gt.events
    assert "chair_1_reentry_frame" in gt.events

    nodes = graph.query_nodes(concept="chair")
    assert len(nodes) == 1, "re-entering object must bind to the existing node, not create a new one"
    assert nodes[0].visible is True
    assert nodes[0].observation_count > 1


def test_ttc_matches_analytic_ground_truth_on_crossing_hazard():
    _, gt = crossing_hazard(speed_mps=1.0, duration_s=1.0)
    # edge_radius_m is a proximity gate unrelated to what this test checks (the
    # TTC formula), so it's widened well past the ~12.7m gap still open after
    # a 1s scenario -- otherwise the edge is filtered out before TTC runs.
    graph = SpatialMemoryGraph(edge_radius_m=50.0)
    _run_scenario(graph, gt)

    node = graph.query_nodes(concept="bicycle")[0]
    last_frame = gt.frames[-1]
    wearer_pos = last_frame.pose.t_world_cam
    wearer_vel = np.array([0.0, 0.0, WALKING_SPEED])

    edges = graph.edges(wearer_pos, wearer_vel)
    assert len(edges) == 1
    computed_ttc = edges[0].ttc_s

    true_object_pos = gt.objects[0].position_world(last_frame.t)
    true_object_vel = gt.objects[0].velocity_world
    analytic_cs = closing_speed(wearer_pos, wearer_vel, true_object_pos, true_object_vel)
    analytic_range = float(np.linalg.norm(true_object_pos - wearer_pos))
    analytic_ttc = time_to_collision(analytic_range, analytic_cs)

    assert computed_ttc == pytest.approx(analytic_ttc, rel=0.05)


def test_mask_lifting_beats_box_lifting_on_synthetic_occlusion():
    frames, gt = cluttered_occlusion(duration_s=0.1)
    gt_frame = gt.frames[0]
    frame = frames[0]
    assert set(gt_frame.object_visible.values()) == {True}, "both chairs should be visible"

    intrinsics = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    errors = {"mask": [], "box": []}

    for obj_id, mask in gt_frame.object_mask.items():
        box_norm = gt_frame.object_box_norm[obj_id]
        detection = Detection(
            concept="chair", instance_id=1, mask=mask, box_xywh_norm=box_norm, score=1.0
        )
        true_depth = next(
            o.position_world(gt_frame.t)[2] for o in gt.objects if o.object_id == obj_id
        )

        mask_inst = lift(detection, frame.depth, intrinsics)
        box_inst = lift_from_box(detection, frame.depth, intrinsics)
        assert mask_inst is not None and box_inst is not None

        errors["mask"].append(abs(mask_inst.p_cam[2] - true_depth))
        errors["box"].append(abs(box_inst.p_cam[2] - true_depth))

    assert sum(errors["mask"]) <= sum(errors["box"])
