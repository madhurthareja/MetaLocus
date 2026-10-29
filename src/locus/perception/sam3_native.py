"""SAM 3.1 Object Multiplex via the native sam3 package (sam3.model_builder),
run locally on the DGX Spark.

Deliberately NOT a single multi-concept-in-one-pass session: the native
SAM 3.1 Object Multiplex checkpoint has no multi-prompt path, so this
backend owns one session per concept (each up to ~128 instances of that
one concept), 4-6 live concept slots chosen by the ConceptProposer. See
docs/PROJECT_PLAN.md section 5 for why (SAM 3.1's efficiency gain is
multiplexing instances of one concept, not multiple concepts at once --
that's only possible via the SAM 3 HF Transformers multi-prompt path,
which doesn't carry the 3.1 Object Multiplex checkpoint).

Stub tonight: requires the gated facebook/sam3.1 checkpoint and the
vendored sam3 package, neither of which this module should assume are
present at import time.
"""

from __future__ import annotations

from locus.types import Detection, Frame


class Sam31NativeSegmenter:
    def __init__(self, checkpoint_repo: str = "facebook/sam3.1", device: str = "cpu"):
        self.checkpoint_repo = checkpoint_repo
        self.device = device
        self._sessions: dict[str, object] = {}  # concept -> native session handle

    def start(self, concepts: list[str]) -> None:
        raise NotImplementedError

    def update_concepts(self, concepts: list[str]) -> None:
        """Start sessions for newly-added concepts, close sessions for dropped ones."""
        raise NotImplementedError

    def step(self, frame: Frame) -> list[Detection]:
        """Propagate every live concept session to this frame and merge their outputs."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
