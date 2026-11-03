"""Spark-narrated hazard directives -- the warm, "friend walking with you"
voice for anything that ISN'T a split-second emergency.

Real constraint this respects: a seconds-away hazard needs a short, fast
spoken warning with zero network round-trip in the loop --
`directives.describe()` handles that tier unconditionally (deterministic,
local, no API call), and is also this module's own fallback if a call here
fails. Everything else -- an object a few seconds out, or just nearby and
not closing -- has real margin, so spending one Meta Model API call
(muse-spark-1.3, the same model and endpoint `spark_proposer.py` already
uses for concept proposal) to turn the geometric facts into a natural
sentence instead of a robotic "concept at clock" readout costs nothing
safety-wise: `generate_dashboard_data.py` is a batch script, not a live
loop, so this call's latency only affects how long *generation* takes,
never the resulting spoken clip's playback duration.

Checked the Meta Model API is actually the right place for this before
writing it: Muse Glimmer -- the model `docs/PROJECT_PLAN.md` originally
scoped for this always-on narration role -- is explicitly self-host-only
(confirmed against https://dev.meta.ai/docs/models: its API column reads
"None (self-host)", and it's absent from `GET /v1/models`'s hosted list).
Muse Spark is hosted and already proven in this codebase, so it's the
real substitute, not Glimmer.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from locus.actuation.directives import clock_hour, describe
from locus.types import Edge

META_API_BASE_URL = "https://api.meta.ai/v1"
MODEL_ID = "muse-spark-1.3"


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

    def narrate(self, concept: str, edge: Edge) -> str:
        """One natural sentence describing `edge`, in a companion's voice.
        Falls back to the fast templated describe() on any API failure --
        a busy walk shouldn't go silent because of a network hiccup."""
        try:
            client = self._client_lazy()
            clock = clock_hour(edge.bearing_rad)
            closing = edge.ttc_s is not None
            facts = (
                f"a {concept} is about {edge.range_m:.0f} meters away at the "
                f"{clock} o'clock position"
                + (f", slowly closing, roughly {edge.ttc_s:.0f} seconds out" if closing else ", not approaching")
            )
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
                            "unless they're truly necessary. Calm, warm, conversational tone."
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
