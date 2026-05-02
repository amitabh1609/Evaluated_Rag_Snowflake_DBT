"""Unit tests for fixed and semantic chunkers."""

import pytest


def test_fixed_chunker_respects_token_limit():
    """Each chunk from the fixed chunker must be <= 512 tokens."""
    pytest.skip("Implement in Phase 2")


def test_fixed_chunker_overlap_is_present():
    """Adjacent chunks should share the expected overlap tokens."""
    pytest.skip("Implement in Phase 2")


def test_semantic_chunker_produces_nonempty_chunks():
    """Semantic chunker should produce at least one chunk for any non-empty input."""
    pytest.skip("Implement in Phase 2")


def test_chunk_metadata_includes_doc_id_and_index():
    """Every chunk must carry doc_id and chunk_index in its metadata."""
    pytest.skip("Implement in Phase 2")
