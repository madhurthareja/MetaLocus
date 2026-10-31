"""Muse Spark 1.3 concept proposer, via Meta's Model API (dev.meta.ai).

The Model API is OpenAI-compatible (confirmed against the actual docs at
https://dev.meta.ai/docs/protocols/chat-completions and
https://dev.meta.ai/docs/image-understanding, not assumed): base URL
https://api.meta.ai/v1, bearer auth, standard `openai` SDK, image input as
a base64 data URI in a user message's content array, and structured output
via `response_format` / `client.beta.chat.completions.parse()`.

This is the one piece of the hackathon's own sanctioned Meta Model API
this project actually calls -- everything else real (SAM 3.1, SeamlessM4T)
is a local open-weights model.
"""

from __future__ import annotations

import base64
import os

from pydantic import BaseModel

from locus.types import Frame

META_API_BASE_URL = "https://api.meta.ai/v1"
MODEL_ID = "muse-spark-1.3"


class _ConceptProposal(BaseModel):
    concepts: list[str]


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
