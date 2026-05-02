"""Test that the generator refuses to answer when context is empty or irrelevant."""

import pytest


def test_generator_declines_when_no_context():
    """Generator must return a 'cannot answer' response, not a hallucination."""
    pytest.skip("Implement in Phase 3")
