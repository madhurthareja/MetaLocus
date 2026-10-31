"""Processes an ENTIRE real sequence (every frame, not a window) through
the full real pipeline -- Spark concept selection, SAM 3.1, lifting,
persistent memory graph, TTC, directives, real TTS -- and dumps everything
a frontend needs to replay it: per-frame images (undistorted + mask
overlay), a JSON timeline of graph state over time, and one spoken-audio
clip per distinct directive (cached by text, not regenerated per frame).

This is the data-generation half of the dashboard; `dashboard/app.py`
serves what this writes.

Usage: python scripts/generate_dashboard_data.py --dataset sanpo [session_id]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from locus.actuation.directives import describe
from locus.actuation.tts import build_tts_backend
from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.geometry.frames import ego_to_world
from locus.geometry.kinematics import finite_difference_velocity
from locus.geometry.lifting import lift
from locus.memory.graph import SpatialMemoryGraph
from locus.perception.sam3_native import Sam31NativeSegmenter
from locus.types import Instance3D

DASHBOARD_IMG_WIDTH = 640


def _resize(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = DASHBOARD_IMG_WIDTH / w
    return cv2.resize(rgb, (DASHBOARD_IMG_WIDTH, int(h * scale)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_id", nargs="?", default=None)
    parser.add_argument("--dataset", choices=["aea", "sanpo"], default="sanpo")
    parser.add_argument("--concepts", default=None, help="comma-separated; auto-picked via Spark if omitted")
    parser.add_argument("--edge-radius-m", type=float, default=20.0)
    parser.add_argument("--out", default=None, help="output dir, default dashboard/data/<dataset>_<session>")
    parser.add_argument("--speak", action="store_true", default=True)
    parser.add_argument("--no-speak", dest="speak", action="store_false")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)

    if args.dataset == "sanpo":
        from locus.data.sanpo_loader import load_sequence

        session_id = args.sequence_id or config["sanpo_sequences"][0]
        print(f"[1/6] Loading FULL SANPO session {session_id!r}...")
        seq = load_sequence(session_id, config=config)
        print(f"      {seq.n_frames} frames @ {seq.fps}fps = {seq.n_frames / seq.fps:.1f}s")
        frames = list(seq.iter_frames())
    else:
        from locus.data.aea_loader import load_point_cloud, load_sequence

        session_id = args.sequence_id or config["aea_sequences"][0]
        print(f"[1/6] Loading FULL AEA sequence {session_id!r}...")
        seq = load_sequence(session_id, config=config)
        points_world = load_point_cloud(session_id, config=config)
        frames = list(seq.iter_frames(point_cloud_world=points_world))

    coverage = np.mean([float((f.depth > 0).mean()) for f in frames])
    print(f"      mean depth coverage: {coverage * 100:.1f}% of pixels")

    out_dir = Path(args.out or f"dashboard/data/{args.dataset}_{session_id}")
    frames_dir, masks_dir, audio_dir = out_dir / "frames", out_dir / "masks", out_dir / "audio"
    for d in (frames_dir, masks_dir, audio_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        print(f"[2/6] Using fixed concepts: {concepts}")
    else:
        from locus.perception.spark_proposer import SparkProposer

        print("[2/6] Proposing concepts from the first frame (Meta Model API, muse-spark-1.3)...")
        concepts = SparkProposer(max_concepts=config["perception"]["max_concepts"]).propose(frames[0], {})
        print(f"      Spark proposed: {concepts}")

    palette = [(220, 60, 60), (60, 160, 220), (80, 200, 100), (220, 170, 50), (160, 90, 210), (60, 200, 200)]
    concept_colors = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

    sam_frame_dir = Path(tempfile.mkdtemp(prefix="locus_sam31_full_"))
    try:
        print(f"[3/6] Running real SAM 3.1 Object Multiplex on {len(concepts)} concepts across all {len(frames)} frames...")
        segmenter = Sam31NativeSegmenter()
        segmenter.start(concepts)
        segmenter.prepare(frames, sam_frame_dir)
    finally:
        shutil.rmtree(sam_frame_dir, ignore_errors=True)

    print("[4/6] Lifting detections, feeding the persistent memory graph, rendering per-frame output...")
    graph = SpatialMemoryGraph(
        node_decay_half_life_seconds=config["memory"]["node_decay_half_life_seconds"],
        node_eviction_seconds=config["memory"]["node_eviction_seconds"],
        edge_radius_m=args.edge_radius_m,
    )
    tts = build_tts_backend(config) if args.speak else None
    audio_cache: dict[str, str] = {}  # directive text -> wav filename

    timeline = []
    prev_pose = None
    wearer_velocity = np.zeros(3)
    total_detections = total_lifted = 0

    for i, frame in enumerate(frames):
        detections = segmenter.step(frame)
        total_detections += len(detections)

        if prev_pose is not None:
            dt = frame.pose.timestamp - prev_pose.timestamp
            if dt > 0:
                wearer_velocity = finite_difference_velocity(prev_pose.t_world_cam, frame.pose.t_world_cam, dt)
        prev_pose = frame.pose

        instances = []
        for det in detections:
            inst_cam = lift(det, frame.depth, seq.intrinsics)
            if inst_cam is None:
                continue
            p_world = ego_to_world(inst_cam.p_cam, frame.pose)
            instances.append(
                Instance3D(
                    concept=inst_cam.concept, instance_id=inst_cam.instance_id, p_cam=inst_cam.p_cam,
                    p_world=p_world, extent=inst_cam.extent, score=inst_cam.score,
                )
            )
        total_lifted += len(instances)
        graph.observe(instances, frame.pose, frame.timestamp)
        graph.step(frame.timestamp)

        edges = graph.edges(frame.pose.t_world_cam, wearer_velocity)
        edge_records = []
        for edge in edges:
            node = graph.nodes[edge.dst_node_id]
            directive = describe(node.concept, edge)
            if tts is not None and directive not in audio_cache:
                fname = f"{hashlib.sha1(directive.encode()).hexdigest()[:10]}.wav"
                tts.synthesize_to_file(directive, str(audio_dir / fname))
                audio_cache[directive] = fname
            edge_records.append(
                {
                    "node_id": edge.dst_node_id,
                    "concept": node.concept,
                    "range_m": round(edge.range_m, 2),
                    "bearing_deg": round(float(np.degrees(edge.bearing_rad)), 1),
                    "ttc_s": round(edge.ttc_s, 2) if edge.ttc_s is not None else None,
                    "confidence": round(node.confidence, 2),
                    "directive": directive,
                    "audio": audio_cache.get(directive),
                }
            )

        rgb_small = _resize(frame.rgb)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

        vis = frame.rgb.copy()
        for det in detections:
            color = concept_colors.get(det.concept, (200, 200, 200))
            overlay = vis.copy()
            overlay[det.mask] = color
            vis = (0.5 * vis + 0.5 * overlay).astype(np.uint8)
        vis_small = _resize(vis)
        cv2.imwrite(str(masks_dir / f"{i:05d}.jpg"), cv2.cvtColor(vis_small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

        timeline.append(
            {
                "frame": i,
                "t": round(frame.timestamp, 3),
                "wearer_pos": [round(float(v), 3) for v in frame.pose.t_world_cam],
                "edges": edge_records,
            }
        )

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"      {i + 1}/{len(frames)} frames processed, {len(graph.nodes)} nodes live")

    segmenter.close()

    print("[5/6] Writing timeline.json...")
    manifest = {
        "dataset": args.dataset,
        "session_id": session_id,
        "fps": seq.fps,
        "n_frames": len(frames),
        "concepts": concepts,
        "edge_radius_m": args.edge_radius_m,
        "total_detections": total_detections,
        "total_lifted": total_lifted,
        "final_summary": graph.summary(),
        "timeline": timeline,
    }
    (out_dir / "timeline.json").write_text(json.dumps(manifest))

    print(f"[6/6] Done. Wrote {len(frames)} frame/mask pairs, {len(audio_cache)} unique spoken clips to {out_dir}")
    print(f"      Total raw detections: {total_detections}, lifted to 3D: {total_lifted}")
    print(f"      Final graph: {graph.summary()}")


if __name__ == "__main__":
    main()
