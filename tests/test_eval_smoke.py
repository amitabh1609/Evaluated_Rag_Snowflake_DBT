"""Smoke tests for the evaluation harness (no LLM calls, no running Qdrant)."""

from pathlib import Path

import pytest
import yaml


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "src" / "erag" / "eval" / "benchmark.yaml"
SMOKE_IDS = {"f-001", "f-003", "f-005", "f-010", "c-001", "c-005", "a-001", "adv-001", "adv-003", "adv-009"}
TIER_COUNTS = {"factual": 15, "conceptual": 15, "architectural": 10, "adversarial": 10}


def _load() -> list[dict]:
    return yaml.safe_load(BENCHMARK_PATH.read_text())


def test_benchmark_loads():
    questions = _load()
    assert len(questions) == 50


def test_benchmark_tier_distribution():
    questions = _load()
    counts = {}
    for q in questions:
        counts[q["tier"]] = counts.get(q["tier"], 0) + 1
    assert counts == TIER_COUNTS


def test_benchmark_no_duplicate_ids():
    questions = _load()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate IDs found"


def test_benchmark_required_fields():
    questions = _load()
    for q in questions:
        for field in ("id", "tier", "question", "gold_answer", "must_cite_doc_ids"):
            assert field in q, f"missing field '{field}' in question {q.get('id', '?')}"
        assert isinstance(q["must_cite_doc_ids"], list) and len(q["must_cite_doc_ids"]) > 0


def test_smoke_ids_all_present():
    questions = _load()
    present = {q["id"] for q in questions}
    missing = SMOKE_IDS - present
    assert not missing, f"smoke IDs missing from benchmark: {missing}"


def test_ragas_runner_imports():
    """Verify the ragas_runner module can be imported without errors."""
    from erag.eval.ragas_runner import (
        SMOKE_IDS as runner_smoke_ids,
        _load_benchmark,
        METRICS_ORDER,
        TIER_ORDER,
    )
    assert len(METRICS_ORDER) == 4
    assert len(TIER_ORDER) == 4
    assert runner_smoke_ids == SMOKE_IDS


def test_load_benchmark_smoke_filter():
    from erag.eval.ragas_runner import _load_benchmark

    questions = _load_benchmark(smoke=True)
    assert len(questions) == len(SMOKE_IDS)
    for q in questions:
        assert q["id"] in SMOKE_IDS


def test_load_benchmark_tier_filter():
    from erag.eval.ragas_runner import _load_benchmark

    adversarial = _load_benchmark(tier="adversarial")
    assert len(adversarial) == 10
    assert all(q["tier"] == "adversarial" for q in adversarial)
