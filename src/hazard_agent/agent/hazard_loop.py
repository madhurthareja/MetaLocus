"""Muse Glimmer 30B, running as an always-on local hazard-monitoring agent.

Continuously consumes the scene graph + TTC values and emits a hazard
event -> TTS warning when a threshold is crossed. This loop runs
independent of any user query (contrast with agent/query.py, which only
answers on request).
"""


class HazardLoop:
    def __init__(self, scene_graph, ttc_threshold_seconds: float, tts_backend):
        self.scene_graph = scene_graph
        self.ttc_threshold_seconds = ttc_threshold_seconds
        self.tts_backend = tts_backend

    def step(self, frame_index: int):
        """Check all tracked nodes' TTC against the threshold and speak a warning if crossed."""
        raise NotImplementedError
