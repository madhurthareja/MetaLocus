"""Muse Spark 1.3 concept proposer, via Meta's Model API (dev.meta.ai).

The Model API is OpenAI-compatible per https://dev.meta.ai/docs/protocols/
chat-completions: base URL https://api.meta.ai/v1, bearer auth, the
standard `openai` SDK, image input as a base64 data URI, and structured
output via `response_format`/`client.beta.chat.completions.parse()`.
"""

from __future__ import annotations

import base64
import os

from typing import Literal

from pydantic import BaseModel

from locus.types import Frame

META_API_BASE_URL = "https://api.meta.ai/v1"
MODEL_ID = "muse-spark-1.3"


class _ConceptProposal(BaseModel):
    concepts: list[str]


class _MotionClassification(BaseModel):
    concept: str
    motion_class: Literal["static", "movable"]
    # Required by the schema either way; read only when motion_class is
    # "movable" -- a model asked for a nullable-only-sometimes field tends
    # to fill it in regardless, so the caller enforces the "static -> None"
    # rule itself rather than trusting this to be null for static concepts.
    max_speed_mps: float


class _MotionClassifications(BaseModel):
    classifications: list[_MotionClassification]


class SparkProposer:
    def __init__(self, max_concepts: int = 6, api_key: str | None = None):
        self.max_concepts = max_concepts
        self._api_key = api_key
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            key = self._api_key or os.environ.get("META_KEY")
            if not key:
                raise RuntimeError(
                    "META_KEY not set -- add it to .env (see .env.example) or pass api_key="
                )
            self._client = OpenAI(base_url=META_API_BASE_URL, api_key=key)
        return self._client

    def propose(self, frame: Frame, graph_summary: dict) -> list[str]:
        import cv2

        client = self._client_lazy()

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for Spark")
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        live_concepts = ", ".join(graph_summary.get("concepts_live", [])) or "none yet"
        response = client.beta.chat.completions.parse(
            model=MODEL_ID,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "You select which object categories an assistive navigation system "
                        "should track for a visually impaired pedestrian, from the current "
                        f"camera frame. Propose up to {self.max_concepts} short, generic "
                        "noun-phrase concepts (e.g. 'car', 'chair', 'person') worth segmenting "
                        "right now -- nearby obstacles and things in or near the direction of "
                        f"travel. Concepts already being tracked ({live_concepts}) should be "
                        "kept if still relevant, so tracking doesn't needlessly churn."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Propose concepts to track in this frame."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
            response_format=_ConceptProposal,
        )
        parsed = response.choices[0].message.parsed
        return parsed.concepts[: self.max_concepts]

    def classify_motion(self, concepts: list[str]) -> dict[str, tuple[str, float | None]]:
        """For concepts locus.memory.motion_classes' hardcoded vocabulary
        doesn't cover: ask whether each is static (never moves on its own)
        or movable, and if movable, a generous real-world top speed for
        the category, in an outdoor pedestrian setting. One call for the
        whole batch, not one per concept -- this runs whenever a new
        concept enters the tracked vocabulary, not per frame.

        No image: this is a question about the concept in general, not
        this specific frame, and keeping it text-only means it can run
        immediately when a concept first appears rather than waiting for
        another camera frame.
        """
        client = self._client_lazy()
        response = client.beta.chat.completions.parse(
            model=MODEL_ID,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "For each given object category, decide: can an "
                        "individual instance of it move under its own power or "
                        "be driven/ridden (movable), or does it only ever sit "
                        "in place once installed (static)? Err toward movable "
                        "when genuinely unsure -- it is safer to watch "
                        "something that turns out not to move than to ignore "
                        "something that does. For every category, also give "
                        "max_speed_mps: a generous (not average) top speed in "
                        "meters/second a single instance could realistically "
                        "reach outdoors near a pedestrian -- for a static "
                        "category this field is still required by the "
                        "response schema but is ignored by the caller, so "
                        "answer it plainly rather than agonizing over it."
                    ),
                },
                {"role": "user", "content": "Classify: " + ", ".join(concepts)},
            ],
            response_format=_MotionClassifications,
        )
        parsed = response.choices[0].message.parsed
        return {
            c.concept: (c.motion_class, c.max_speed_mps if c.motion_class == "movable" else None)
            for c in parsed.classifications
        }
