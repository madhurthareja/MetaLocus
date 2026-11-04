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


def test_duplicate_instance_ids_merge_once_they_converge_spatially():
    """Confirmed happening for real: SAM 3.1's multiplex tracker can emit
    two different instance_ids for one physical object before its own
    de-duplication converges (seen on a real AEA clip -- two overlapping
    chair masks, 0.29m apart after lifting, that never merged because the
    association gate is only checked at node-creation time). Here the two
    instances start > the 1.0m gate apart (so they're forced into separate
    nodes) and then converge to within it on a later observation."""
    graph = SpatialMemoryGraph(association_gate_m=1.0)
    inst_a = Instance3D(
        concept="chair", instance_id=100, p_cam=np.zeros(3),
        p_world=np.array([0.0, 0.0, 5.0]), extent=np.zeros(3), score=1.0,
    )
    inst_b = Instance3D(
        concept="chair", instance_id=200, p_cam=np.zeros(3),
        p_world=np.array([0.0, 0.0, 6.5]), extent=np.zeros(3), score=1.0,
    )
    graph.observe([inst_a, inst_b], None, t=0.0)
    assert len(graph.nodes) == 2, "should start as two separate nodes -- the gate miss is the whole setup"

    inst_a2 = Instance3D(
        concept="chair", instance_id=100, p_cam=np.zeros(3),
        p_world=np.array([0.0, 0.0, 5.0]), extent=np.zeros(3), score=1.0,
    )
    inst_b2 = Instance3D(
        concept="chair", instance_id=200, p_cam=np.zeros(3),
        p_world=np.array([0.0, 0.0, 5.2]), extent=np.zeros(3), score=1.0,
    )
    graph.observe([inst_a2, inst_b2], None, t=1.0)

    assert len(graph.nodes) == 1, "nodes within the gate after a later observation should merge"
    merged_node = next(iter(graph.nodes.values()))
    assert merged_node.observation_count == 4, "merged node should carry both tracks' observation history"

    # Both original instance_ids must now resolve to the single surviving node
    # (otherwise the next observation of either would spuriously recreate a node).
    node_ids = {graph._instance_to_node[100], graph._instance_to_node[200]}
    assert node_ids == {merged_node.node_id}


def test_static_object_jitter_produces_smaller_velocity_than_raw_finite_difference():
    """Real bug, found on the real SANPO session: node.velocity_world was a
    raw single-step finite difference, so ordinary mask/depth-lifting jitter
    on a genuinely static object (a street pole) produced spurious
    velocities of 1.4-5.3 m/s. compute_edges()'s closing_speed() then read
    that as real approach, producing "critical" hazard alerts for poles
    sitting to the side of the wearer's path (median bearing ~87 degrees on
    the false-critical edges) rather than ahead of it -- a noise-
    amplification bug, not a threshold-tuning one. Smoothing must bring a
    jittery-but-stationary object's estimated speed well below both a real
    walking pace and what the old raw-difference formula would have given
    for the same noisy positions."""
    graph = SpatialMemoryGraph()
    base = np.array([3.0, 0.0, 4.0])
    rng = np.random.default_rng(0)
    dt = 1 / 15  # matches real SANPO's frame rate
    positions = [base + rng.normal(scale=0.1, size=3) for _ in range(12)]

    raw_speeds = []
    for i, p in enumerate(positions):
        inst = Instance3D(
            concept="pole", instance_id=1, p_cam=np.zeros(3),
            p_world=p, extent=np.zeros(3), score=1.0,
        )
        graph.observe([inst], None, t=float(i) * dt)
        if i > 0:
            raw_speeds.append(float(np.linalg.norm((p - positions[i - 1]) / dt)))

    node = next(iter(graph.nodes.values()))
    smoothed_speed = float(np.linalg.norm(node.velocity_world))
    avg_raw_speed = float(np.mean(raw_speeds))

    assert smoothed_speed < avg_raw_speed * 0.6, (
        f"smoothed speed ({smoothed_speed:.2f} m/s) should be well below the raw "
        f"per-step finite-difference average ({avg_raw_speed:.2f} m/s) for a jittery-but-static object"
    )
    assert smoothed_speed < 1.0, f"a static object should not read as moving at near-walking pace, got {smoothed_speed:.2f} m/s"


def test_sustained_motion_still_converges_despite_smoothing():
    """Smoothing must not hide a genuinely approaching object -- a steady
    velocity should still be recovered within a handful of real
    observations, not permanently damped toward zero."""
    graph = SpatialMemoryGraph()
    true_velocity = np.array([0.0, 0.0, -2.0])  # 2 m/s, genuinely closing
    pos = np.array([0.0, 0.0, 10.0])
    dt = 1 / 15
    for i in range(15):
        inst = Instance3D(
            concept="van", instance_id=1, p_cam=np.zeros(3),
            p_world=pos.copy(), extent=np.zeros(3), score=1.0,
        )
        graph.observe([inst], None, t=float(i) * dt)
        pos = pos + true_velocity * dt

    node = next(iter(graph.nodes.values()))
    recovered_speed = float(np.linalg.norm(node.velocity_world))
    assert recovered_speed > 1.5, f"sustained real motion should still be recovered after smoothing, got {recovered_speed:.2f} m/s"


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


def _observe_track(graph, concept, positions, dt=1 / 15, instance_id=1):
    for i, p in enumerate(positions):
        graph.observe(
            [Instance3D(concept=concept, instance_id=instance_id, p_cam=np.zeros(3),
                        p_world=np.asarray(p, dtype=float), extent=np.zeros(3), score=1.0)],
            None, t=float(i) * dt,
        )
        graph.step(float(i) * dt)
    return next(iter(graph.nodes.values()))


def test_static_concept_velocity_is_pinned_to_zero_under_jitter():
    """Static things don't move: measured lifting noise on real SANPO static
    nodes (median 0.14m, p90 0.70m per frame) made smoothed velocity ~1-13
    m/s and drove ~94% of critical alerts. Pinning, not filtering."""
    rng = np.random.default_rng(1)
    positions = [np.array([3.0, 0.0, 4.0]) + rng.normal(scale=0.4, size=3) for _ in range(20)]
    node = _observe_track(SpatialMemoryGraph(), "street lamp", positions)
    assert np.all(node.velocity_world == 0.0)


def test_mover_velocity_does_not_survive_an_identity_jump():
    """The real SANPO failure: the segmenter re-attaches an instance id to
    a different object (a distant, moving truck's id landing on a parked one
    at 6m). The old object's velocity must not carry over to the new one --
    that inherited speed was behind every false critical mover alert there."""
    dt = 1 / 15
    moving = [np.array([0.0, 0.0, 30.0 - 8.0 * i * dt]) for i in range(20)]
    parked = [np.array([2.0, 0.0, 6.0])] * 12
    graph = SpatialMemoryGraph()
    node = _observe_track(graph, "truck", moving)
    assert np.linalg.norm(node.velocity_world) == pytest.approx(8.0, rel=0.05)

    for i, p in enumerate(parked, start=len(moving)):
        graph.observe([Instance3D(concept="truck", instance_id=1, p_cam=np.zeros(3), p_world=p,
                                  extent=np.zeros(3), score=1.0)], None, t=i * dt)
        if i == len(moving):
            assert np.linalg.norm(node.velocity_world) == 0.0, "the jump starts a new track with no speed"
    assert node.p_world == pytest.approx(parked[-1])
    assert np.linalg.norm(node.velocity_world) < 0.1, "the parked truck's own track says it isn't moving"


def test_mover_velocity_fit_recovers_speed_through_depth_jitter():
    dt = 1 / 15
    rng = np.random.default_rng(0)
    positions = [np.array([0.0, 0.0, 20.0 - 5.0 * i * dt]) + rng.normal(0, 0.15, 3) for i in range(30)]
    graph = SpatialMemoryGraph()
    _observe_track(graph, "car", positions[:5])
    assert np.linalg.norm(graph.nodes[1].velocity_world) == 0.0, "no speed until the track spans enough time"
    node = _observe_track(graph, "car", positions)
    assert node.velocity_world[2] == pytest.approx(-5.0, rel=0.15)


def test_mover_flipping_between_two_objects_never_gets_a_speed():
    """One id alternating between two objects 15m apart every frame (seen on
    real SANPO frames 90-93) is all jumps -- it must read as no motion,
    not as a vehicle doing 200 m/s or anything averaged from that."""
    positions = [np.array([0.0, 0.0, 6.0 if i % 2 else 21.0]) for i in range(30)]
    node = _observe_track(SpatialMemoryGraph(), "truck", positions)
    assert np.linalg.norm(node.velocity_world) == 0.0


def test_unknown_concept_keeps_class_agnostic_velocity():
    dt = 1 / 15
    positions = [np.array([0.0, 0.0, 10.0 - 30.0 * i * dt]) for i in range(15)]
    node = _observe_track(SpatialMemoryGraph(), "puddle", positions)
    assert np.linalg.norm(node.velocity_world) > 20.0, "no class prior -> no speed ceiling"


def test_stale_movable_node_gets_no_edge_but_static_node_does():
    """A remembered car is a guess that gets worse every frame; a remembered
    pole is exactly right. Both stay in memory, only the pole keeps an edge."""
    graph = SpatialMemoryGraph(edge_radius_m=20.0)
    for concept, iid, p in (("car", 1, [2.0, 0.0, 6.0]), ("pole", 2, [-1.0, 0.0, 4.0])):
        graph.observe([Instance3D(concept=concept, instance_id=iid, p_cam=np.zeros(3),
                                  p_world=np.array(p), extent=np.zeros(3), score=1.0)], None, t=0.0)
    wearer, walking = np.zeros(3), np.array([0.0, 0.0, 1.2])

    graph.step(0.5)
    assert {graph.nodes[e.dst_node_id].concept for e in graph.edges(wearer, walking, t=0.5)} == {"car", "pole"}

    graph.step(1.5)
    assert {graph.nodes[e.dst_node_id].concept for e in graph.edges(wearer, walking, t=1.5)} == {"pole"}
    assert {n.concept for n in graph.nodes.values()} == {"car", "pole"}, "stale mover is still remembered"
    assert {graph.nodes[e.dst_node_id].concept for e in graph.edges(wearer, walking)} == {"car", "pole"}, (
        "without t, no staleness filter (old call sites unchanged)"
    )


def test_movable_node_decays_and_evicts_faster_than_static():
    graph = SpatialMemoryGraph(node_decay_half_life_seconds=8.0, node_eviction_seconds=20.0)
    for concept, iid, p in (("truck", 1, [2.0, 0.0, 6.0]), ("tree", 2, [-1.0, 0.0, 4.0])):
        graph.observe([Instance3D(concept=concept, instance_id=iid, p_cam=np.zeros(3),
                                  p_world=np.array(p), extent=np.zeros(3), score=1.0)], None, t=0.0)
    graph.step(2.0)
    by_concept = {n.concept: n for n in graph.nodes.values()}
    assert by_concept["truck"].confidence < 0.3 < 0.8 < by_concept["tree"].confidence
    graph.step(5.0)
    assert {n.concept for n in graph.nodes.values()} == {"tree"}


def test_closest_approach_separates_passing_beside_from_head_on():
    from locus.memory.edges import closest_approach

    walking = np.array([0.0, 0.0, -1.4])  # object's velocity relative to a wearer walking +Z at 1.4 m/s
    cpa, tcpa = closest_approach(np.array([1.0, 0.0, 3.0]), walking)
    assert cpa == pytest.approx(1.0) and tcpa == pytest.approx(3.0 / 1.4)
    cpa, _ = closest_approach(np.array([0.0, 0.0, 3.0]), walking)
    assert cpa == pytest.approx(0.0)
    cpa, tcpa = closest_approach(np.array([1.0, 0.0, -3.0]), walking)  # already behind
    assert cpa == pytest.approx(np.hypot(1.0, 3.0)) and tcpa is None


def test_mover_velocity_reports_unmeasured_until_the_track_is_long_enough():
    dt = 1 / 15
    graph = SpatialMemoryGraph()
    positions = [np.array([0.0, 0.0, 15.0 - 1.0 * i * dt]) for i in range(12)]
    _observe_track(graph, "bicycle", positions[:3])
    assert not graph.velocity_measured(1), "zero here means unknown, not stopped"
    _observe_track(graph, "bicycle", positions)
    assert graph.velocity_measured(1)
    _observe_track(graph, "street lamp", [np.zeros(3)], instance_id=2)
    assert graph.velocity_measured(2), "a static concept's zero is a measurement"
