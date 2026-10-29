"""Muse Wonder query-answering layer over the scene graph.

Answers free-form spatial questions (used both for the benchmark eval
in eval/ and for ad-hoc demo Q&A) by querying the persistent scene graph
rather than reasoning over the current frame alone.
"""


class QueryEngine:
    def __init__(self, scene_graph):
        self.scene_graph = scene_graph

    def answer(self, anchor: str, relation: str, target: str) -> str:
        raise NotImplementedError
