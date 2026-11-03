"""Real SAM 3.1 Object Multiplex backend.

Native path constraint (this corrects chat.md's original, incorrect
assumption that a single SAM 3 session could take multiple concurrent
concept prompts): the checkpoint's video predictor supports exactly one
text-prompted concept per session. Live use means one session per live
concept slot (4-6 per PROJECT_PLAN.md), each rerunning the Perception
Encoder.

A second, more basic mismatch with the Segmenter protocol: SAM 3.1's video
API (`start_session` / `add_prompt` / `propagate_in_video` / `close_session`)
is batch, not streaming -- you point a session at a materialized directory
of frames and propagate through all of them once, you don't feed it one
frame at a time. So this class has a `prepare()` step that does the actual
model work up front (materializes frames to disk, runs one
start/prompt/propagate/close session per concept), and `step()` just
replays the already-computed per-frame result in order. That's a real
architectural difference from MockSegmenter (which is genuinely streaming)
worth knowing before reading this as "the same shape as the mock."
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from locus.types import Detection, Frame


class Sam31NativeSegmenter:
    def __init__(
        self,
        version: str = "sam3.1",
        checkpoint_path: str | None = None,
        max_num_objects: int = 16,
        compile: bool = False,
        use_fa3: bool = False,
    ):
        self._version = version
        self._checkpoint_path = checkpoint_path
        self._max_num_objects = max_num_objects
        self._compile = compile
        # Flash Attention 3 needs the separate `flash_attn_interface`
        # package (not a plain pip install, often a platform-specific
        # build); off by default so this runs correctly without it, at
        # some speed cost. Flip on once FA3 is actually set up on GB10.
        self._use_fa3 = use_fa3
        self._predictor = None
        self._frame_dir: Path | None = None
        self._active_concepts: list[str] = []
        self._detections_by_frame: dict[int, list[Detection]] = {}
        self._cursor = 0
        # SAM's out_obj_ids are only unique *within* a session (one per
        # concept), so two concepts both yield an obj_id of 0 for their
        # first tracked instance -- remap to a globally-unique id or the
        # memory graph's instance_id -> node_id map would collide them.
        self._global_instance_ids: dict[tuple[str, int], int] = {}
        self._next_global_instance_id = 1

    def start(self, concepts: list[str]) -> None:
        self._active_concepts = list(concepts)
        self._cursor = 0

    def _build_predictor(self):
        if self._predictor is None:
            from sam3 import build_sam3_predictor

            self._predictor = build_sam3_predictor(
                version=self._version,
                checkpoint_path=self._checkpoint_path,
                max_num_objects=self._max_num_objects,
                compile=self._compile,
                use_fa3=self._use_fa3,
                async_loading_frames=False,
            )
        return self._predictor

    @staticmethod
    def _start_session(predictor, resource_path: str) -> str:
        """Replicates Sam3BasePredictor.start_session, minus one keyword
        (`offload_state_to_cpu`) that this checkout's base predictor passes
        unconditionally but Sam3MultiplexTrackingWithInteractivity.init_state
        doesn't accept -- a version-skew bug in the vendored sam3 repo, not
        something to patch in someone else's cloned library on a shared
        machine. `handle_request({"type": "start_session", ...})` hits that
        bug directly; calling `model.init_state` ourselves and registering
        the session exactly like the base class does sidesteps it."""
        inference_state = predictor.model.init_state(resource_path=resource_path)
        session_id = str(uuid.uuid4())
        predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id

    def prepare(self, frames: list[Frame], frame_dir: str | Path) -> None:
        """Does the real model work: writes `frames` to `frame_dir` as
        sequentially-numbered JPGs (SAM 3.1's video loader expects a frame
        directory, not individual arrays), then runs one
        start_session/add_prompt/propagate_in_video/close_session cycle per
        active concept, merging results across concepts into a per-frame
        Detection list. Call `start()` first to set the concept list.

        Kept as whole-video propagation (start_frame_idx=0, direction=both's
        forward half) for the concepts known at session start, same as
        before -- `add_concept_at()` below handles concepts discovered
        later, propagating forward-only from the frame they were noticed."""
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), bgr)

        self._frame_dir = frame_dir
        self._predictor = self._build_predictor()
        self._detections_by_frame = {i: [] for i in range(len(frames))}

        for concept in self._active_concepts:
            self._run_concept_session(concept, start_frame_idx=0)

    def add_concept_at(self, concept: str, start_frame_idx: int) -> bool:
        """Give a concept its own SAM 3.1 session starting at `start_frame_idx`
        instead of frame 0 -- for a concept Muse Spark only noticed partway
        through the walk (the one-shot, frame-0-only proposal this replaces
        could never see anything that wasn't visible in the opening frame).

        Deliberately NOT a fresh full-video session: `propagate_in_video`
        natively supports `start_frame_idx` + `propagation_direction` (see
        sam3_base_predictor.py), so this only tracks forward from where the
        concept was actually noticed, same cost shape as any other
        single-concept session, just starting later and running less of the
        video. Returns False (and runs no propagation) if SAM 3.1 finds
        nothing for this text at that frame -- Spark proposing a concept
        doesn't guarantee it's actually there.
        """
        if self._predictor is None or self._frame_dir is None:
            raise RuntimeError("add_concept_at() requires prepare() to have run first")
        return self._run_concept_session(concept, start_frame_idx=start_frame_idx)

    def ground_text(self, concept: str, frame: Frame) -> tuple[float, float, float, float] | None:
        """One-shot text->box grounding on a single frame -- no propagation,
        no dependency on prepare()'s multi-frame directory. This is the
        "ground once" half of the real-time split discussed for glasses
        hardware: EdgeTamSegmenter uses this as its default grounder,
        trading SAM 3.1's whole-clip batch cost for a single cheap add_prompt
        call per concept, then hands off to EdgeTAM for genuinely causal,
        per-frame tracking. Returns None if SAM 3.1 finds nothing here --
        same "proposed but not actually present" case add_concept_at()
        already has to handle.
        """
        predictor = self._build_predictor()
        with tempfile.TemporaryDirectory(prefix="sam31_ground_") as tmp:
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(tmp) / "00000.jpg"), bgr)
            session_id = self._start_session(predictor, tmp)
            try:
                result = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": concept,
                    }
                )
                boxes = np.asarray(result["outputs"]["out_boxes_xywh"])
                if len(boxes) == 0:
                    return None
                return tuple(float(v) for v in boxes[0])
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})

    def _run_concept_session(self, concept: str, start_frame_idx: int) -> bool:
        session_id = self._start_session(self._predictor, str(self._frame_dir))
        found = False
        try:
            add_result = self._predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": start_frame_idx,
                    "text": concept,
                }
            )
            if len(np.asarray(add_result["outputs"]["out_obj_ids"])) == 0:
                return False  # Spark proposed this, but SAM 3.1 sees nothing here -- don't pay for propagation
            found = True
            for result in self._predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": start_frame_idx,
                    "propagation_direction": "forward",
                }
            ):
                frame_idx = result["frame_index"]
                outputs = result["outputs"]
                obj_ids = np.asarray(outputs["out_obj_ids"])
                masks = np.asarray(outputs["out_binary_masks"])
                boxes = np.asarray(outputs["out_boxes_xywh"])
                for obj_id, mask, box in zip(obj_ids, masks, boxes):
                    key = (concept, int(obj_id))
                    if key not in self._global_instance_ids:
                        self._global_instance_ids[key] = self._next_global_instance_id
                        self._next_global_instance_id += 1
                    self._detections_by_frame.setdefault(frame_idx, []).append(
                        Detection(
                            concept=concept,
                            instance_id=self._global_instance_ids[key],
                            mask=mask.astype(bool),
                            box_xywh_norm=tuple(float(v) for v in box),
                            score=1.0,
                        )
                    )
        finally:
            self._predictor.handle_request({"type": "close_session", "session_id": session_id})
        return found

    def step(self, frame: Frame) -> list[Detection]:
        detections = self._detections_by_frame.get(self._cursor, [])
        self._cursor += 1
        return detections

    def close(self) -> None:
        self._predictor = None
