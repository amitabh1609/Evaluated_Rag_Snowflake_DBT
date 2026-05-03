"""Fixed-size chunking: 512 tokens, 64-token overlap (cl100k_base tokeniser).

Used for Config A (baseline) and Config B in the ablation.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from erag.chunk import Chunk

_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter(chunk_size: int = 512, chunk_overlap: int = 64) -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None or _splitter._chunk_size != chunk_size:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        _splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter


def chunk_document(
    doc_id: str,
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split `text` into fixed-size token chunks.

    Args:
        doc_id: Unique document identifier from manifest.
        text: Raw document text (markdown).
        metadata: Passthrough payload (title, url, source, …).
        chunk_size: Token limit per chunk.
        chunk_overlap: Overlap in tokens between adjacent chunks.

    Returns:
        Ordered list of Chunk objects.
    """
    splitter = _get_splitter(chunk_size, chunk_overlap)
    parts = splitter.split_text(text)
    meta = metadata or {}
    total = len(parts)
    return [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}:{i}",
            text=part,
            chunk_index=i,
            total_chunks=total,
            metadata={**meta, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
        for i, part in enumerate(parts)
    ]
