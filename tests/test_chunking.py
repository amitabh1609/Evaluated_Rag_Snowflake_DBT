"""Unit tests for fixed and semantic chunkers."""

import pytest

from erag.chunk import Chunk
from erag.chunk.fixed import chunk_document as fixed_chunk


SAMPLE_TEXT = """
# Snowflake Virtual Warehouses

A virtual warehouse is a named abstraction for a cluster of compute resources in Snowflake.
Warehouses are available in several sizes, from X-Small to 6X-Large.

## Auto-Suspend

When a warehouse is inactive for a specified period, Snowflake automatically suspends it to
stop consuming credits. The default auto-suspend setting created via the UI is 600 seconds
(10 minutes). When creating a warehouse via SQL, AUTO_SUSPEND must be set explicitly.

## Auto-Resume

Auto-resume automatically starts a suspended warehouse when a SQL statement is submitted.
It is enabled by default for all warehouses.
""".strip()


def test_fixed_chunker_returns_chunks():
    chunks = fixed_chunk("test_doc", SAMPLE_TEXT)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_fixed_chunker_chunk_ids_are_sequential():
    chunks = fixed_chunk("test_doc", SAMPLE_TEXT)
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"test_doc:{i}"
        assert c.chunk_index == i
        assert c.total_chunks == len(chunks)


def test_fixed_chunker_no_empty_chunks():
    chunks = fixed_chunk("test_doc", SAMPLE_TEXT)
    for c in chunks:
        assert c.text.strip(), f"chunk {c.chunk_id} is empty"


def test_fixed_chunker_metadata_passthrough():
    meta = {"title": "Warehouses", "url": "https://docs.snowflake.com/en/user-guide/warehouses"}
    chunks = fixed_chunk("test_doc", SAMPLE_TEXT, metadata=meta)
    for c in chunks:
        assert c.metadata["title"] == "Warehouses"
        assert c.metadata["url"] == meta["url"]


def test_fixed_chunker_long_doc_splits():
    long_text = SAMPLE_TEXT * 20  # well over 512 tokens
    chunks = fixed_chunk("long_doc", long_text, chunk_size=512)
    assert len(chunks) > 1, "long document should produce multiple chunks"


def test_fixed_chunker_custom_sizes():
    chunks_small = fixed_chunk("doc", SAMPLE_TEXT, chunk_size=64, chunk_overlap=8)
    chunks_large = fixed_chunk("doc", SAMPLE_TEXT, chunk_size=512, chunk_overlap=64)
    # smaller chunk size → more chunks
    assert len(chunks_small) >= len(chunks_large)


def test_fixed_chunker_empty_string():
    chunks = fixed_chunk("empty_doc", "")
    # Empty input: no chunks or one empty chunk — either is acceptable
    for c in chunks:
        assert c.doc_id == "empty_doc"


@pytest.mark.skip(reason="requires tokenizers package at test time — run manually after uv sync")
def test_semantic_chunker_returns_nonempty_chunks():
    from erag.chunk.semantic import chunk_document as semantic_chunk

    chunks = semantic_chunk("test_doc", SAMPLE_TEXT)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip()


@pytest.mark.skip(reason="requires tokenizers package at test time — run manually after uv sync")
def test_semantic_chunker_metadata_passthrough():
    from erag.chunk.semantic import chunk_document as semantic_chunk

    meta = {"source": "snowflake"}
    chunks = semantic_chunk("test_doc", SAMPLE_TEXT, metadata=meta)
    for c in chunks:
        assert c.metadata["source"] == "snowflake"
