"""Smoke test: run RAGAS on the 10-question CI subset and assert faithfulness >= threshold."""

import pytest


def test_smoke_faithfulness_above_threshold():
    """Run the 10-question smoke eval; fail if faithfulness < 0.80."""
    pytest.skip("Implement in Phase 4 — also used by GitHub Actions eval gate")
