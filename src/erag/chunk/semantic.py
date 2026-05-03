"""Semantic chunking: split at embedding-similarity boundaries.

Used for Config C (primary) in the ablation.
semantic-text-splitter uses a HuggingFace tokenizer to respect the token
budget and finds split points where sentence-embedding similarity drops.
"""

from __future__ import annotations

from erag.chunk import Chunk

_splitter = None  # lazy-initialised — loading the tokenizer takes ~1 s


def _get_splitter(max_tokens: int = 512):
    global _splitter
    if _splitter is None:
        from semantic_text_splitter import TextSplitter
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        _splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, capacity=max_tokens)
    return _splitter


def chunk_document(
    doc_id: str,
    text: str,
    metadata: dict | None = None,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Split `text` at semantic boundaries up to `max_tokens` per chunk.

    Args:
        doc_id: Unique document identifier from manifest.
        text: Raw document text (markdown).
        metadata: Passthrough payload.
        max_tokens: Maximum tokens per chunk.

    Returns:
        Ordered list of Chunk objects.
    """
    splitter = _get_splitter(max_tokens)
    parts = splitter.chunks(text)
    meta = metadata or {}
    total = len(parts)
    return [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}:{i}",
            text=part,
            chunk_index=i,
            total_chunks=total,
            metadata={**meta, "max_tokens": max_tokens, "chunker": "semantic"},
        )
        for i, part in enumerate(parts)
    ]
