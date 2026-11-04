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
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from locus.actuation.directives import describe
from locus.actuation.narrator import SparkNarrator
from locus.actuation.speech_policy import SpeechPolicy
from locus.actuation.tts import build_tts_backend
from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.geometry.frames import WORLD_DOWN_DEFAULT, ego_to_world, horizontal
from locus.geometry.kinematics import WearerMotion
from locus.geometry.lifting import lift
from locus.memory.graph import SpatialMemoryGraph
from locus.memory.motion_classes import DEFAULT_CAUTIOUS_MAX_SPEED_MPS, MOVABLE, UNKNOWN, motion_class, register_motion_classification
from locus.types import Instance3D

DASHBOARD_IMG_WIDTH = 640
N_CONCEPT_CHECKPOINTS = 3  # later re-proposal points through the clip, beyond the initial frame-0 one
MAX_NEW_CONCEPTS_PER_CHECKPOINT = 2  # bounds extra SAM 3.1 sessions per checkpoint


def classify_unlisted_concepts(concepts: list[str]) -> None:
    """Ask Spark to classify any concept locus.memory.motion_classes'
    hardcoded static/movable vocabulary doesn't already cover, so it
    isn't left with zero velocity protection for the rest of the run --
    the hardcoded lists can't anticipate every concept Spark proposes.
    Costs one Spark call per newly-seen unlisted concept, not per frame;
    a concept already classified in a prior call this run is skipped.
    """
    unlisted = [c for c in dict.fromkeys(concepts) if motion_class(c) == UNKNOWN]
    if not unlisted:
        return
    print(f"       {len(unlisted)} concept(s) outside the known static/movable vocabulary {unlisted}: asking Spark...")
    from locus.perception.spark_proposer import SparkProposer

    try:
        classifications = SparkProposer().classify_motion(unlisted)
        for concept in unlisted:
            kind, speed = classifications.get(concept, (MOVABLE, DEFAULT_CAUTIOUS_MAX_SPEED_MPS))
            register_motion_classification(concept, kind, speed)
            print(f"         {concept!r} -> {kind}" + (f" (max {speed:.0f} m/s)" if kind == MOVABLE else ""))
    except Exception as exc:
        # A failed classification call must not mean "run unprotected" --
        # that's the exact gap this function exists to close. Default to
        # movable/cautious (DEFAULT_CAUTIOUS_MAX_SPEED_MPS) instead of
        # leaving these concepts at the old unconstrained behavior.
        print(f"         Spark classification failed ({exc}); defaulting to movable/cautious rather than unprotected")
        for concept in unlisted:
            register_motion_classification(concept, MOVABLE, DEFAULT_CAUTIOUS_MAX_SPEED_MPS)


def _resize(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = DASHBOARD_IMG_WIDTH / w
    return cv2.resize(rgb, (DASHBOARD_IMG_WIDTH, int(h * scale)))


def _encode_session_video(image_dir: Path, out_path: Path, fps: float) -> None:
    """The dashboard used to fetch every frame/mask as its own JPEG over
    HTTP -- hundreds of uncompressed requests per session, which a
    browser's 6-connections-per-origin cap and any real network latency
    turn into stuttering playback. Muxing the same JPEGs into one H.264
    file instead cuts that to a single request with real compression and
    lets the browser's native <video> buffering do the work.

    -g 1 (every frame an independent keyframe) trades away inter-frame
    compression so the dashboard's frame-accurate scrubbing (a seek per
    timeline tick during playback) stays cheap -- seeking into the middle
    of a long GOP would otherwise have to decode back to its keyframe
    first.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(image_dir / "%05d.jpg"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-g", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    from locus.memory.motion_classes import reset_registered_classifications

    reset_registered_classifications()  # a fresh registry per run, not per process

    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_id", nargs="?", default=None)
    parser.add_argument("--dataset", choices=["aea", "sanpo", "synthetic"], default="sanpo")
    parser.add_argument("--scenario", default="crossing_hazard", help="synthetic scenario name (--dataset synthetic only)")
    parser.add_argument("--concepts", default=None, help="comma-separated; auto-picked via Spark if omitted (ignored for synthetic)")
    parser.add_argument(
        "--segmenter",
        choices=["sam3.1", "edgetam"],
        default="sam3.1",
        help="sam3.1: whole-clip batch propagation, highest accuracy (default). "
        "edgetam: genuinely causal/streaming (no look-ahead), the real-time-capable "
        "path -- grounds each concept via one one-shot SAM 3.1 add_prompt call, "
        "then tracks live frame-by-frame with EdgeTAM (ignored for synthetic, which "
        "always uses MockSegmenter).",
    )
    parser.add_argument("--edge-radius-m", type=float, default=20.0)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="truncate real (sanpo/aea) sequences to this many frames -- AEA sessions run "
        "1999-4262 frames at 20fps vs SANPO's ~400-540 at 15fps, far too long for a batch run "
        "at full length. Ignored for synthetic, which is already short.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="first frame index for real (sanpo/aea) sequences")
    parser.add_argument("--out", default=None, help="output dir, default dashboard/data/<dataset>_<session>")
    parser.add_argument("--speak", action="store_true", default=True)
    parser.add_argument("--no-speak", dest="speak", action="store_false")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)

    ground_truth = None
    if args.dataset == "sanpo":
        from locus.data.sanpo_loader import load_sequence

        session_id = args.sequence_id or config["sanpo_sequences"][0]
        print(f"[1/7] Loading FULL SANPO session {session_id!r}...")
        seq = load_sequence(session_id, config=config)
        print(f"      {seq.n_frames} frames @ {seq.fps}fps = {seq.n_frames / seq.fps:.1f}s")
        print("      prefetching all RGB+depth files in parallel (16 workers)...")
        seq.prefetch()
        frames = list(seq.iter_frames(max_frames=args.max_frames, start_frame=args.start_frame))
    elif args.dataset == "aea":
        from locus.data.aea_loader import load_point_cloud, load_sequence

        session_id = args.sequence_id or config["aea_sequences"][0]
        print(f"[1/7] Loading AEA sequence {session_id!r}...")
        seq = load_sequence(session_id, config=config)
        print(f"      {seq.n_video_frames} frames @ {seq.fps}fps = {seq.n_video_frames / seq.fps:.1f}s total"
              + (f", using {args.max_frames} from frame {args.start_frame}" if args.max_frames else ""))
        points_world = load_point_cloud(session_id, config=config)
        frames = list(seq.iter_frames(
            point_cloud_world=points_world, max_frames=args.max_frames, start_frame=args.start_frame,
        ))
    else:
        from locus.data import synthetic

        session_id = args.scenario
        print(f"[1/7] Generating synthetic scenario {session_id!r} (deterministic ground truth, no GPU)...")
        scenario_fn = getattr(synthetic, args.scenario)
        frames, ground_truth = scenario_fn()
        print(f"      {len(frames)} frames @ {synthetic.FPS}fps")

        class _FakeSeq:
            intrinsics = synthetic.DEFAULT_INTRINSICS
            fps = synthetic.FPS

        seq = _FakeSeq()

    coverage = np.mean([float((f.depth > 0).mean()) for f in frames])
    print(f"      mean depth coverage: {coverage * 100:.1f}% of pixels")

    out_dir = Path(args.out or f"dashboard/data/{args.dataset}_{session_id}")
    frames_dir, masks_dir, audio_dir = out_dir / "frames", out_dir / "masks", out_dir / "audio"
    for d in (frames_dir, masks_dir, audio_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    if ground_truth is not None:
        concepts = sorted({obj.concept for obj in ground_truth.objects})
        print(f"[2/7] Concepts from synthetic ground truth: {concepts}")
    elif args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        print(f"[2/7] Using fixed concepts: {concepts}")
    else:
        from locus.perception.spark_proposer import SparkProposer

        print("[2/7] Proposing concepts from the first frame (Meta Model API, muse-spark-1.3)...")
        concepts = SparkProposer(max_concepts=config["perception"]["max_concepts"]).propose(frames[0], {})
        print(f"      Spark proposed: {concepts}")

    if ground_truth is None:
        classify_unlisted_concepts(concepts)

    palette = [(220, 60, 60), (60, 160, 220), (80, 200, 100), (220, 170, 50), (160, 90, 210), (60, 200, 200)]
    concept_colors = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

    sam_frame_dir = Path(tempfile.mkdtemp(prefix="locus_sam31_full_"))
    try:
        if ground_truth is not None:
            from locus.perception.mock_segmenter import MockSegmenter

            print(f"[3/7] Using MockSegmenter (ground-truth-driven, no model) on concepts {concepts}...")
            segmenter = MockSegmenter(ground_truth)
            segmenter.start(concepts)
        elif args.segmenter == "sam3.1":
            from locus.perception.sam3_native import Sam31NativeSegmenter

            print(f"[3/7] Running real SAM 3.1 Object Multiplex on {len(concepts)} concepts across all {len(frames)} frames...")
            segmenter = Sam31NativeSegmenter()
            segmenter.start(concepts)
            segmenter.prepare(frames, sam_frame_dir)

            # The one-shot proposal above only ever sees the opening frame, so
            # anything not visible yet at t=0 (a dog, scaffolding, a scooter
            # two minutes in) was permanently outside the vocabulary for the
            # rest of the walk -- Muse Spark was never asked again. Re-ask at
            # a few later checkpoints and give any genuinely new concept its
            # own forward-only session (add_concept_at, not a fresh
            # whole-video prepare()) starting where it was actually noticed.
            # N_CONCEPT_CHECKPOINTS stays small deliberately: each new
            # concept costs one more SAM 3.1 session re-running the
            # Perception Encoder, and this is a one-time batch generation
            # run, not the live ~0.5Hz refresh PROJECT_PLAN.md describes for
            # an always-on system -- that cadence (every ~2s) would mean
            # dozens of extra sessions per clip, not a handful.
            #
            # Skipped when --concepts fixed the vocabulary explicitly: that's
            # an intentional override, not something to second-guess, and
            # SparkProposer was never imported on that path.
            if not args.concepts:
                from locus.perception.spark_proposer import SparkProposer

                print(f"[3b/7] Re-checking for new concepts at {N_CONCEPT_CHECKPOINTS} later checkpoints...")
                checkpoint_proposer = SparkProposer(max_concepts=config["perception"]["max_concepts"])
                for k in range(1, N_CONCEPT_CHECKPOINTS + 1):
                    idx = min(len(frames) - 1, round(len(frames) * k / (N_CONCEPT_CHECKPOINTS + 1)))
                    proposed = checkpoint_proposer.propose(frames[idx], {})
                    new_candidates = [c for c in proposed if c not in concepts][:MAX_NEW_CONCEPTS_PER_CHECKPOINT]
                    added_this_checkpoint = []
                    for concept in new_candidates:
                        if segmenter.add_concept_at(concept, start_frame_idx=idx):
                            concepts.append(concept)
                            concept_colors[concept] = palette[len(concept_colors) % len(palette)]
                            added_this_checkpoint.append(concept)
                            print(f"       +{concept!r} at t={frames[idx].timestamp:.1f}s (frame {idx})")
                        else:
                            print(f"       skipped {concept!r} at frame {idx} -- Spark proposed it, SAM 3.1 found nothing there")
                    classify_unlisted_concepts(added_this_checkpoint)
        else:
            from locus.perception.edgetam_segmenter import EdgeTamSegmenter

            print(
                f"[3/7] Running EdgeTAM (causal/streaming, no look-ahead) on {len(concepts)} "
                f"concepts across all {len(frames)} frames -- grounded via one-shot SAM 3.1 "
                "text->box calls as each concept is first encountered..."
            )
            segmenter = EdgeTamSegmenter()
            segmenter.start(concepts)

            # EdgeTAM's step() already tries to ground any not-yet-grounded
            # active concept on every frame it's given (see
            # edgetam_segmenter.py) -- unlike SAM 3.1's one-shot, whole-clip
            # proposal, it doesn't need a special "start tracking from here"
            # call. So widening the vocabulary at the same later checkpoints
            # SAM 3.1 uses is just appending to the active-concepts list;
            # EdgeTAM picks each one up the first frame it's actually visible.
            if not args.concepts:
                from locus.perception.spark_proposer import SparkProposer

                print(f"[3b/7] Re-checking for new concepts at {N_CONCEPT_CHECKPOINTS} later checkpoints...")
                checkpoint_proposer = SparkProposer(max_concepts=config["perception"]["max_concepts"])
                for k in range(1, N_CONCEPT_CHECKPOINTS + 1):
                    idx = min(len(frames) - 1, round(len(frames) * k / (N_CONCEPT_CHECKPOINTS + 1)))
                    proposed = checkpoint_proposer.propose(frames[idx], {})
                    new_candidates = [c for c in proposed if c not in concepts][:MAX_NEW_CONCEPTS_PER_CHECKPOINT]
                    if new_candidates:
                        concepts.extend(new_candidates)
                        for concept in new_candidates:
                            concept_colors[concept] = palette[len(concept_colors) % len(palette)]
                        segmenter.update_concepts(concepts)
                        classify_unlisted_concepts(new_candidates)
                        print(f"       widened vocabulary with {new_candidates} at t={frames[idx].timestamp:.1f}s (frame {idx})")
    finally:
        shutil.rmtree(sam_frame_dir, ignore_errors=True)

    print("[4/7] Lifting detections, feeding the persistent memory graph, rendering per-frame output...")
    world_down = np.asarray(getattr(seq, "world_down", WORLD_DOWN_DEFAULT), dtype=float)
    graph = SpatialMemoryGraph(
        node_decay_half_life_seconds=config["memory"]["node_decay_half_life_seconds"],
        node_eviction_seconds=config["memory"]["node_eviction_seconds"],
        edge_radius_m=args.edge_radius_m,
        world_down=world_down,
    )
    tts = build_tts_backend(config) if args.speak else None
    # Narration only matters if something will actually speak it; gated on
    # --speak the same way `tts` is, so --no-speak never pays for the API
    # calls either.
    narrator = SparkNarrator() if args.speak else None
    audio_cache: dict[str, str] = {}  # directive text -> wav filename
    node_extent: dict[int, np.ndarray] = {}  # node_id -> most recent observed extent (Node has no extent field)
    node_events: list[dict] = []  # a running log of create/re-associate/merge/evict, for World Memory's event log

    timeline = []
    wearer_motion = WearerMotion(world_down=world_down)
    total_detections = total_lifted = 0
    seen_node_ids: set[int] = set()
    speech = SpeechPolicy()

    for i, frame in enumerate(frames):
        detections = segmenter.step(frame)
        total_detections += len(detections)

        wearer_velocity, wearer_heading = wearer_motion.update(
            frame.timestamp, frame.pose.t_world_cam, camera_forward_world=frame.pose.R_world_cam[:, 2]
        )

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

        was_visible = {nid: n.visible for nid, n in graph.nodes.items()}
        prev_last_seen = {nid: n.last_seen for nid, n in graph.nodes.items()}
        graph.observe(instances, frame.pose, frame.timestamp)
        for inst in instances:
            node_id = graph._instance_to_node.get(inst.instance_id)
            if node_id is not None:
                node_extent[node_id] = inst.extent
        for node_id, node in graph.nodes.items():
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                node_events.append(
                    {"t": round(frame.timestamp, 3), "frame": i, "kind": "node_created",
                     "node_id": node_id, "concept": node.concept}
                )
            elif was_visible.get(node_id) is False and node.visible:
                node_events.append(
                    {"t": round(frame.timestamp, 3), "frame": i, "kind": "reassociated",
                     "node_id": node_id, "concept": node.concept,
                     "out_of_view_s": round(frame.timestamp - prev_last_seen.get(node_id, frame.timestamp), 2)}
                )
        graph.step(frame.timestamp)

        edges = graph.edges(frame.pose.t_world_cam, wearer_velocity, wearer_heading, t=frame.timestamp)

        # At most one edge is spoken per frame, and only when it's news and
        # the channel is free (see locus.actuation.speech_policy). Every
        # edge still gets a directive string and a record either way, for
        # the dashboard's own display of full tracked state, but only the
        # elected one gets an audio clip attached.
        speak_node_id = speech.select(edges, frame.timestamp)

        edge_records = []
        for edge in edges:
            node = graph.nodes[edge.dst_node_id]
            # The warm, narrated voice only ever applies to the one edge
            # actually chosen to speak this frame, and only for a routine
            # (advisory) mention -- anything on a collision course always
            # gets the short, local, no-API-call describe() path instead,
            # deliberately: a narrated sentence takes ~1s longer to speak,
            # and an approaching object's lead time can't spare it. The
            # severity used is the speech policy's (after its critical
            # hold), so the timeline/HUD agree with the audio. Every other (non-spoken)
            # edge also uses describe() here, purely as cheap display text
            # for the UI's tables -- not worth an API call for something
            # that's never actually going to be spoken.
            edge_severity = speech.severities[edge.dst_node_id]
            if narrator is not None and edge.dst_node_id == speak_node_id and edge_severity == "advisory":
                speed = (
                    float(np.linalg.norm(horizontal(node.velocity_world, world_down)))
                    if graph.velocity_measured(edge.dst_node_id) else None
                )
                directive = narrator.narrate(node.concept, edge, object_speed_mps=speed)
            else:
                directive = describe(node.concept, edge)
            audio = None
            if edge.dst_node_id == speak_node_id:
                if tts is not None and directive not in audio_cache:
                    fname = f"{hashlib.sha1(directive.encode()).hexdigest()[:10]}.wav"
                    tts.synthesize_to_file(directive, str(audio_dir / fname))
                    audio_cache[directive] = fname
                audio = audio_cache.get(directive)
            edge_records.append(
                {
                    "node_id": edge.dst_node_id,
                    "concept": node.concept,
                    "range_m": round(edge.range_m, 2),
                    "bearing_deg": round(float(np.degrees(edge.bearing_rad)), 1),
                    "ttc_s": round(edge.ttc_s, 2) if edge.ttc_s is not None else None,
                    "cpa_m": round(edge.cpa_m, 2) if edge.cpa_m is not None else None,
                    "severity": edge_severity,
                    "spoken": edge.dst_node_id == speak_node_id,
                    "confidence": round(node.confidence, 2),
                    "directive": directive,
                    "audio": audio,
                }
            )

        # Full persistent-memory node state -- every live node, not just those
        # within edge_radius_m -- is what the 3D World View needs to show
        # objects that are remembered but currently out of hazard range too.
        node_records = []
        for node_id, node in graph.nodes.items():
            extent = node_extent.get(node_id, np.array([0.3, 0.3, 0.3]))
            node_records.append(
                {
                    "node_id": node_id,
                    "concept": node.concept,
                    "p_world": [round(float(v), 3) for v in node.p_world],
                    "extent": [round(float(v), 3) for v in extent],
                    "velocity_world": [round(float(v), 3) for v in node.velocity_world],
                    "visible": bool(node.visible),
                    "confidence": round(float(node.confidence), 3),
                    "first_seen": round(float(node.first_seen), 3),
                    "last_seen": round(float(node.last_seen), 3),
                    "observation_count": node.observation_count,
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
                "wearer_velocity": [round(float(v), 3) for v in wearer_velocity],
                "wearer_heading": None if wearer_heading is None else [round(float(v), 4) for v in wearer_heading],
                "camera_R_world_cam": [[round(float(v), 4) for v in row] for row in frame.pose.R_world_cam],
                "n_detections_this_frame": len(detections),
                "edges": edge_records,
                "nodes": node_records,
            }
        )

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"      {i + 1}/{len(frames)} frames processed, {len(graph.nodes)} nodes live")

    if hasattr(segmenter, "close"):
        segmenter.close()

    print("[5/7] Writing timeline.json...")
    manifest = {
        "dataset": args.dataset,
        "session_id": session_id,
        "fps": seq.fps,
        "n_frames": len(frames),
        "concepts": concepts,
        "edge_radius_m": args.edge_radius_m,
        "world_down": [float(v) for v in world_down],
        "total_detections": total_detections,
        "total_lifted": total_lifted,
        "final_summary": graph.summary(),
        "node_events": node_events,
        "timeline": timeline,
    }
    (out_dir / "timeline.json").write_text(json.dumps(manifest))

    print("[6/7] Encoding frames.mp4 / masks.mp4...")
    _encode_session_video(frames_dir, out_dir / "frames.mp4", seq.fps)
    _encode_session_video(masks_dir, out_dir / "masks.mp4", seq.fps)

    print(f"[7/7] Done. Wrote {len(frames)} frame/mask pairs, {len(audio_cache)} unique spoken clips to {out_dir}")
    print(f"      Total raw detections: {total_detections}, lifted to 3D: {total_lifted}")
    print(f"      Final graph: {graph.summary()}")


if __name__ == "__main__":
    main()
