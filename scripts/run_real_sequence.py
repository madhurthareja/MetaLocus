"""Runs the complete real-data thread on one real sequence -- AEA (indoor)
or SANPO (outdoor): real pose, real RGB, real depth, real SAM 3.1
segmentation, through the persistent memory graph, out to audio cues and
spoken directives.

This is a one-shot script, not `app/pipeline.py` (the eventual
productionized async version) -- it exists to prove the full real thread
actually works end-to-end before that's built. Both datasets converge on
the same `locus.types.Frame` (rgb, depth, pose), which is why one script
drives either: everything from SAM 3.1 onward is dataset-agnostic.

Usage:
  python scripts/run_real_sequence.py --dataset aea [sequence_id] [--n-frames N] [--concepts a,b,c]
  python scripts/run_real_sequence.py --dataset sanpo [session_id] [--n-frames N] [--concepts a,b,c]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from locus.actuation.directives import describe
from locus.actuation.tts import build_tts_backend
from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.geometry.frames import bearing_clockface, world_to_body
from locus.geometry.kinematics import finite_difference_velocity
from locus.geometry.lifting import lift
from locus.memory.graph import SpatialMemoryGraph
from locus.perception.sam3_native import Sam31NativeSegmenter
from locus.types import Instance3D


def _load_frames_aea(sequence_id, config, args):
    from locus.data.aea_loader import load_point_cloud, load_sequence

    print(f"[1/5] Loading AEA sequence {sequence_id!r} (pose + fisheye-undistorted RGB)...")
    seq = load_sequence(sequence_id, config=config)

    print("[2/5] Loading + filtering semi-dense point cloud...")
    points_world = load_point_cloud(sequence_id, config=config)
    print(f"      {len(points_world)} points after distance-uncertainty filtering")

    print(f"[3/5] Materializing {args.n_frames} frames with real sparse depth (from frame {args.start_frame})...")
    frames = list(
        seq.iter_frames(
            stride=args.stride, max_frames=args.n_frames, point_cloud_world=points_world,
            start_frame=args.start_frame,
        )
    )
    return seq, frames


def _load_frames_sanpo(session_id, config, args):
    from locus.data.sanpo_loader import load_sequence

    print(f"[1/5] Loading SANPO session {session_id!r} (pose + pinhole RGB)...")
    seq = load_sequence(session_id, config=config)
    print(f"      {seq.n_frames} frames available @ {seq.fps}fps")

    print(f"[2/5] (SANPO ships dense depth directly -- no point-cloud projection needed)")

    print(f"[3/5] Materializing {args.n_frames} frames with real dense depth (from frame {args.start_frame})...")
    frames = list(seq.iter_frames(stride=args.stride, max_frames=args.n_frames, start_frame=args.start_frame))
    return seq, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_id", nargs="?", default=None)
    parser.add_argument("--dataset", choices=["aea", "sanpo"], default="aea")
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--concepts", default=None, help="comma-separated; ignored if --auto-concepts is set")
    parser.add_argument(
        "--auto-concepts", action="store_true",
        help="pick concepts from the first frame via Muse Spark (Meta Model API) instead of a fixed list",
    )
    parser.add_argument("--edge-radius-m", type=float, default=None)
    parser.add_argument("--save-mask-viz", default=None, help="path to save a mask overlay PNG")
    parser.add_argument("--speak", default=None, help="directory to write one spoken-directive .wav per node")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)

    if args.dataset == "aea":
        sequence_id = args.sequence_id or config["aea_sequences"][0]
        seq, frames = _load_frames_aea(sequence_id, config, args)
    else:
        sequence_id = args.sequence_id or config["sanpo_sequences"][0]
        seq, frames = _load_frames_sanpo(sequence_id, config, args)

    if args.auto_concepts:
        from locus.perception.spark_proposer import SparkProposer

        print("[Spark] proposing concepts from the first frame (Meta Model API, muse-spark-1.3)...")
        concepts = SparkProposer(max_concepts=config["perception"]["max_concepts"]).propose(frames[0], {})
        print(f"      Spark proposed: {concepts}")
    else:
        concepts = [c.strip() for c in (args.concepts or "chair,couch,table").split(",") if c.strip()]

    positions = np.array([f.pose.t_world_cam for f in frames])
    path_m = sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
    print(f"      wearer path length over this window: {path_m:.2f}m ({path_m / (len(frames) / seq.fps):.2f} m/s avg)")
    coverage = np.mean([float((f.depth > 0).mean()) for f in frames])
    print(f"      mean depth coverage: {coverage * 100:.1f}% of pixels")

    frame_dir = Path(tempfile.mkdtemp(prefix="locus_sam31_"))
    try:
        print(f"[4/5] Running real SAM 3.1 Object Multiplex on concepts {concepts}...")
        print(f"      (one native session per concept, frames staged at {frame_dir})")
        segmenter = Sam31NativeSegmenter()
        segmenter.start(concepts)
        segmenter.prepare(frames, frame_dir)

        if args.save_mask_viz:
            import cv2

            palette = [(220, 60, 60), (60, 160, 220), (80, 200, 100), (220, 170, 50), (160, 90, 210), (60, 200, 200)]
            concept_colors = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

            # Pick the frame with the most detections instead of always the
            # midpoint -- the midpoint can genuinely have zero of the
            # requested concepts in view (e.g. before a car enters frame).
            viz_frame_idx = max(
                range(len(frames)),
                key=lambda i: len(segmenter._detections_by_frame.get(i, [])),
            )
            vis = frames[viz_frame_idx].rgb.copy()
            dets_at_frame = segmenter._detections_by_frame.get(viz_frame_idx, [])
            for det in dets_at_frame:
                color = concept_colors.get(det.concept, (200, 200, 200))
                overlay = vis.copy()
                overlay[det.mask] = color
                vis = (0.5 * vis + 0.5 * overlay).astype(np.uint8)
            cv2.imwrite(args.save_mask_viz, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"      saved mask overlay (frame {viz_frame_idx}, {len(dets_at_frame)} detections) to {args.save_mask_viz}")

        graph = SpatialMemoryGraph(
            node_decay_half_life_seconds=config["memory"]["node_decay_half_life_seconds"],
            node_eviction_seconds=config["memory"]["node_eviction_seconds"],
            edge_radius_m=args.edge_radius_m or config["memory"]["edge_radius_m"],
        )

        print("[5/5] Lifting detections to 3D, feeding the persistent memory graph...")
        from locus.geometry.frames import ego_to_world

        prev_pose = None
        wearer_velocity = np.zeros(3)
        total_detections = 0
        total_lifted = 0
        for frame in frames:
            detections = segmenter.step(frame)
            total_detections += len(detections)

            if prev_pose is not None:
                dt = frame.pose.timestamp - prev_pose.timestamp
                if dt > 0:
                    wearer_velocity = finite_difference_velocity(
                        prev_pose.t_world_cam, frame.pose.t_world_cam, dt
                    )
            prev_pose = frame.pose

            instances = []
            for det in detections:
                inst_cam = lift(det, frame.depth, seq.intrinsics)
                if inst_cam is None:
                    continue
                p_world = ego_to_world(inst_cam.p_cam, frame.pose)
                instances.append(
                    Instance3D(
                        concept=inst_cam.concept,
                        instance_id=inst_cam.instance_id,
                        p_cam=inst_cam.p_cam,
                        p_world=p_world,
                        extent=inst_cam.extent,
                        score=inst_cam.score,
                    )
                )
            total_lifted += len(instances)
            graph.observe(instances, frame.pose, frame.timestamp)
            graph.step(frame.timestamp)

        segmenter.close()
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    print()
    print(f"Raw SAM 3.1 detections across all frames: {total_detections}")
    print(f"Successfully lifted to 3D (had enough valid sparse depth): {total_lifted}")
    print()
    print("Final memory graph state:")
    print(f"  {graph.summary()}")

    tts = build_tts_backend(config) if args.speak else None
    if args.speak:
        Path(args.speak).mkdir(parents=True, exist_ok=True)
        print(f"\n[TTS] loading {config['tts']['backend']}...")

    if prev_pose is not None:
        edges = graph.edges(prev_pose.t_world_cam, wearer_velocity)
        print()
        print(f"Edges from the wearer's final position ({len(edges)}):")
        for edge in edges:
            node = graph.nodes[edge.dst_node_id]
            print(
                f"  node {edge.dst_node_id} ({node.concept}): "
                f"range={edge.range_m:.2f}m bearing={np.degrees(edge.bearing_rad):.1f}deg "
                f"ttc={edge.ttc_s} confidence={node.confidence:.2f}"
            )
            directive = describe(node.concept, edge)
            print(f"    directive: \"{directive}\"")
            if tts is not None:
                out_path = str(Path(args.speak) / f"node_{edge.dst_node_id}.wav")
                tts.synthesize_to_file(directive, out_path)
                print(f"    spoken -> {out_path}")


if __name__ == "__main__":
    main()
