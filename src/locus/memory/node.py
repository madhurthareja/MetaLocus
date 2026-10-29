"""Node is defined in locus.types (it's part of the cross-package contract);
this module just holds the node-id allocator so graph.py stays focused on
association/decay/eviction logic.
"""

from __future__ import annotations

from locus.types import Node


class NodeIdAllocator:
    def __init__(self):
        self._next_id = 1

    def allocate(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id


__all__ = ["Node", "NodeIdAllocator"]
