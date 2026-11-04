"""Spark-narrated hazard directives -- the warm, "friend walking with you"
voice for anything that isn't a split-second emergency.

A seconds-away hazard needs a short, fast, local spoken warning with no
network round trip -- `directives.describe()` handles that tier
unconditionally, and is also this module's fallback if a call here
fails. Everything else has real margin, so one Meta Model API call
(muse-spark-1.3) turns the geometric facts into a natural sentence
instead of a "concept at clock" readout; since generation is a batch
script rather than a live loop, this call's latency only affects
generation time, never the resulting clip's playback duration.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from locus.actuation.directives import clock_hour, describe, on_collision_course
from locus.types import Edge

META_API_BASE_URL = "https://api.meta.ai/v1"
MODEL_ID = "muse-spark-1.3"
# Below this, an object reads as stationary -- mover velocity is a line fit
# over noisy lifted positions (see SpatialMemoryGraph), not exact.
MOVING_SPEED_MPS = 0.5


class _Narration(BaseModel):
    text: str


class SparkNarrator:
    def __init__(self, api_key: str | None = None):
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

    def narrate(self, concept: str, edge: Edge, object_speed_mps: float | None = None) -> str:
        """One natural sentence describing `edge`, in a companion's voice.
        Falls back to the fast templated describe() on any API failure --
        a busy walk shouldn't go silent because of a network hiccup."""
        try:
            client = self._client_lazy()
            clock = clock_hour(edge.bearing_rad)
            # State what's actually known about motion rather than leaving
            # the model to guess and fill in something plausible but wrong
            # (or an implied "you're safe" this system has no basis for).
            if on_collision_course(edge):
                motion = f", slowly closing, roughly {edge.ttc_s:.0f} seconds out"
            elif object_speed_mps is not None and object_speed_mps >= MOVING_SPEED_MPS:
                motion = ", moving"
            elif object_speed_mps is not None:
                motion = ", not moving"
            else:
                motion = ""  # speed not measured yet -- say nothing rather than guess
            facts = f"a {concept} is about {edge.range_m:.0f} meters away at the {clock} o'clock position" + motion
            response = client.beta.chat.completions.parse(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a sighted friend walking beside a blind person, "
                            "narrating the scene naturally and warmly as you walk -- not "
                            "reading out sensor data. Respond with exactly one short, "
                            "natural spoken sentence (under 12 words) a real friend would "
                            "actually say out loud. Always mention the clock-face direction "
                            "and the object by name. Skip exact numbers (meters, seconds) "
                            "unless they're truly necessary. Calm, warm, conversational tone. "
                            "Never say or imply that they are safe, that the way is clear, or "
                            "that there's nothing to worry about -- you can only see some "
                            "objects, not curbs, holes or everything else, so describe what "
                            "is there and leave the judgment to them."
                        ),
                    },
                    {"role": "user", "content": f"Narrate this for them: {facts}"},
                ],
                response_format=_Narration,
            )
            text = response.choices[0].message.parsed.text.strip()
            return text or describe(concept, edge)
        except Exception:
            return describe(concept, edge)
