"""Chunking primitives — shared Chunk dataclass used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str        # "{doc_id}:{index}"
    text: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)
    # metadata keys: title, url, source, char_start, char_end

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}:{self.chunk_index}"
