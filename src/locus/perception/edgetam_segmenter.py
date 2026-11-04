"""EdgeTAM -- Meta's distilled, on-device SAM 2 -- as a genuinely streaming
Segmenter backend, for the real-time path (see sam3_native.py's module
docstring for SAM 3.1 Object Multiplex's batch constraint). Unlike
Sam31NativeSegmenter, step() here does real per-frame model work:
`EdgeTamVideoModel.forward(..., frame=pixel_tensor)` registers the new
frame and runs causal inference on it immediately, with no look-ahead
and no materialized frame directory -- the actual candidate for live
glasses hardware, at the cost of lower accuracy and no occlusion
re-identification versus SAM 3.1's batch design.

Loaded via Hugging Face Transformers, not Meta's own EdgeTAM repo --
`yonigozlan/EdgeTAM-hf` is a native `transformers` port (model_type
"edgetam_video") sharing SAM 2's `Sam2VideoProcessor`.

EdgeTAM is SAM 2 lineage: point/box/mask prompts only, no open-vocabulary
text grounding. Something still has to turn a Spark-proposed concept
string into an initial box before EdgeTAM can track it -- that's
`grounder`, a `(concept, frame) -> box_xywh_norm | None` callable,
defaulting to `Sam31NativeSegmenter.ground_text` (one cheap add_prompt
call, no propagation), lazily built on first use so a caller supplying
their own grounder never pays for loading SAM 3.1 at all.

v1 scope: one tracked instance per concept (first grounder match only).
Multi-instance-per-concept would need the grounder to return multiple
boxes and this class to mint one obj_id per box.

Periodic re-grounding (`reground_interval`): a causal tracker with no
reverse pass drifts over a long clip with nothing to correct it. Every
`reground_interval` frames, each already-tracked concept gets one fresh
SAM 3.1 box call on the *current* frame, re-injected for its existing
obj_id -- this lands on a frame_idx the session has never seen before, so
it's treated as a fresh conditioning frame automatically (see step()'s
comment). A transient miss (object briefly occluded) is not treated as
"gone": the existing track is left alone and retried at the next
interval, rather than dropping the object on one failed look.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from locus.types import Detection, Frame

DEFAULT_CHECKPOINT = "yonigozlan/EdgeTAM-hf"

BoxGrounder = Callable[[str, Frame], "tuple[float, float, float, float] | None"]


def _mask_to_box_norm(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    h, w = mask.shape
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return (float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h))


class EdgeTamSegmenter:
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        grounder: BoxGrounder | None = None,
        ground_retry_interval: int = 15,
        reground_interval: int | None = 45,
    ):
        self._checkpoint = checkpoint
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._grounder = grounder
        # A concept not yet visible should keep getting looked for (it may
        # enter frame later), not be given up on after one miss -- but
        # looking means a full SAM 3.1 session, so retrying every frame
        # forever is a real cost bug: a concept that's simply never in
        # this clip would pay for one SAM 3.1 session per frame for the
        # rest of the video. Rate-limit retries instead of removing them.
        self._ground_retry_interval = ground_retry_interval
        # None disables drift correction entirely -- a fixed interval, not
        # confidence-triggered, because object_score_logits dipping is a
        # normal signal even on a well-tracked object, not a reliable
        # "this has drifted" flag.
        self._reground_interval = reground_interval
        self._model = None
        self._processor = None
        self._reshaped_size: tuple[int, int] | None = None
        self._session = None

        self._active_concepts: list[str] = []
        self._grounded_concepts: set[str] = set()
        self._next_ground_attempt_frame: dict[str, int] = {}
        self._next_reground_frame: dict[str, int] = {}
        self._concept_by_obj_id: dict[int, str] = {}
        self._obj_id_by_concept: dict[str, int] = {}
        self._next_obj_id = 1
        self._frame_idx = 0

    def _lazy_load(self):
        if self._model is None:
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor

            self._model = EdgeTamVideoModel.from_pretrained(self._checkpoint).to(self._device).eval()
            self._processor = Sam2VideoProcessor.from_pretrained(self._checkpoint)
            size = self._processor.video_processor.size
            self._reshaped_size = (size["height"], size["width"])

    def _default_grounder(self) -> BoxGrounder:
        if self._grounder is None:
            from locus.perception.sam3_native import Sam31NativeSegmenter

            self._grounder = Sam31NativeSegmenter().ground_text
        return self._grounder

    def start(self, concepts: list[str]) -> None:
        self._lazy_load()
        self._active_concepts = list(concepts)
        self._grounded_concepts = set()
        self._next_ground_attempt_frame = {}
        self._next_reground_frame = {}
        self._concept_by_obj_id = {}
        self._obj_id_by_concept = {}
        self._next_obj_id = 1
        self._frame_idx = 0
        self._session = self._processor.init_video_session(
            inference_device=self._device, dtype=self._dtype
        )

    def update_concepts(self, concepts: list[str]) -> None:
        self._active_concepts = list(concepts)

    def step(self, frame: Frame) -> list[Detection]:
        if self._session is None:
            raise RuntimeError("EdgeTamSegmenter.step() called before start()")

        frame_idx = self._frame_idx
        self._frame_idx += 1
        h, w = frame.rgb.shape[:2]

        pv = self._processor.video_processor(videos=[frame.rgb], device=self._device, return_tensors="pt")
        pixel_tensor = pv.pixel_values_videos[0].to(self._dtype)
        self._session.add_new_frame(pixel_tensor, frame_idx)

        # All new-or-refreshed boxes this frame must go in a single
        # add_inputs_to_inference_session() call: that method *overwrites*
        # (not appends to) inference_session.obj_with_new_inputs, so calling
        # it once per concept would silently drop every earlier object's
        # "has new inputs" flag for this frame and crash inside forward().
        update_obj_ids = []
        update_boxes_xyxy = []
        grounder = self._default_grounder()
        for concept in self._active_concepts:
            if concept not in self._grounded_concepts:
                if frame_idx < self._next_ground_attempt_frame.get(concept, 0):
                    continue
                box_norm = grounder(concept, frame)
                if box_norm is None:
                    self._next_ground_attempt_frame[concept] = frame_idx + self._ground_retry_interval
                    continue
                obj_id = self._next_obj_id
                self._next_obj_id += 1
                self._concept_by_obj_id[obj_id] = concept
                self._obj_id_by_concept[concept] = obj_id
                self._grounded_concepts.add(concept)
                if self._reground_interval:
                    self._next_reground_frame[concept] = frame_idx + self._reground_interval
            elif self._reground_interval and frame_idx >= self._next_reground_frame.get(concept, float("inf")):
                # Scheduled drift-correction check, not a new detection --
                # re-anchor the SAME obj_id with a fresh box. frame_idx here
                # is always a frame this session has never processed before
                # (self._frame_idx only increments), so forward() will treat
                # it as a fresh conditioning frame automatically, the same
                # mechanism a brand-new concept's first grounding uses --
                # no special "reinit" path needed.
                self._next_reground_frame[concept] = frame_idx + self._reground_interval
                box_norm = grounder(concept, frame)
                if box_norm is None:
                    continue  # transient miss -- keep the existing track, don't drop it
                obj_id = self._obj_id_by_concept[concept]
            else:
                continue

            x, y, bw, bh = box_norm
            update_obj_ids.append(obj_id)
            update_boxes_xyxy.append([x * w, y * h, (x + bw) * w, (y + bh) * h])

        if update_obj_ids:
            self._processor.add_inputs_to_inference_session(
                inference_session=self._session,
                frame_idx=frame_idx,
                obj_ids=update_obj_ids,
                input_boxes=[update_boxes_xyxy],
                original_size=(h, w),
            )

        if not self._concept_by_obj_id:
            return []  # nothing grounded yet -- nothing to causally track

        output = self._model(inference_session=self._session, frame_idx=frame_idx)
        masks = self._processor.post_process_masks(
            [output.pred_masks], original_sizes=[(h, w)], reshaped_input_sizes=[self._reshaped_size]
        )[0]  # (num_objects, 1, h, w) bool
        scores = torch.sigmoid(output.object_score_logits).reshape(-1).tolist()

        detections = []
        for i, obj_id in enumerate(output.object_ids):
            concept = self._concept_by_obj_id.get(obj_id)
            if concept is None:
                continue
            mask = masks[i, 0].cpu().numpy().astype(bool)
            detections.append(
                Detection(
                    concept=concept,
                    instance_id=obj_id,
                    mask=mask,
                    box_xywh_norm=_mask_to_box_norm(mask),
                    score=float(scores[i]),
                )
            )
        return detections

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._session = None
