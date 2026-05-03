"""RAGAS evaluation runner for the benchmark.

Loads benchmark.yaml, runs the RAG pipeline for each question, collects
RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall),
and writes per-tier results to JSON and a markdown table.

Usage:
    uv run python -m erag.eval.ragas_runner                    # full 50-question run
    uv run python -m erag.eval.ragas_runner --smoke            # 10-question CI subset
    uv run python -m erag.eval.ragas_runner --output-json /tmp/results.json
    uv run python -m erag.eval.ragas_runner --tier adversarial # single-tier run
    uv run python -m erag.eval.ragas_runner --no-reranker      # ablation helper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from erag.config import PROCESSED_DIR, settings
from erag.generate.answerer import answer as rag_answer

logger = logging.getLogger(__name__)
console = Console()

BENCHMARK_PATH = Path(__file__).parent / "benchmark.yaml"

# 10-question CI smoke subset — spread across tiers
SMOKE_IDS = {"f-001", "f-003", "f-005", "f-010", "c-001", "c-005", "a-001", "adv-001", "adv-003", "adv-009"}

METRICS_ORDER = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
TIER_ORDER = ["factual", "conceptual", "architectural", "adversarial"]


def _load_benchmark(
    tier: str | None = None,
    smoke: bool = False,
    ids: set[str] | None = None,
) -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        questions = yaml.safe_load(f)
    if smoke:
        questions = [q for q in questions if q["id"] in SMOKE_IDS]
    if tier:
        questions = [q for q in questions if q["tier"] == tier]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def _build_ragas_dataset(records: list[dict]):
    """Build a RAGAS-compatible dataset from eval records."""
    try:
        # RAGAS 0.2.x API: EvaluationDataset + SingleTurnSample
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["gold_answer"],
            )
            for r in records
        ]
        return EvaluationDataset(samples=samples)
    except ImportError:
        # Fallback: older RAGAS using HuggingFace Dataset
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "question": [r["question"] for r in records],
                "answer": [r["answer"] for r in records],
                "contexts": [r["contexts"] for r in records],
                "ground_truth": [r["gold_answer"] for r in records],
            }
        )


def _get_metrics():
    """Return RAGAS metric instances, handling API differences across versions."""
    try:
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        return [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    except ImportError:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        return [faithfulness, answer_relevancy, context_precision, context_recall]


def _configure_ragas_llm():
    """Point RAGAS metrics at the configured LLM (Anthropic or OpenAI)."""
    if settings.generator_model.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
            from ragas import adapt

            llm = ChatAnthropic(
                model=settings.generator_model,
                api_key=settings.anthropic_api_key,
            )
            return llm
        except ImportError:
            logger.warning("langchain_anthropic not installed — RAGAS will use default OpenAI LLM")
    return None


def run_eval(
    questions: list[dict],
    use_reranker: bool = True,
    collection: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the RAG pipeline and RAGAS evaluation on a list of benchmark questions.

    Returns:
        (records, results_dict) where records contain per-question data and
        results_dict contains aggregated RAGAS scores.
    """
    records: list[dict] = []
    errors = 0

    console.print(f"\n[bold]Running RAG pipeline on {len(questions)} questions…[/bold]")

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        tier = q["tier"]
        question_text = q["question"]
        gold = q["gold_answer"].strip()

        console.print(f"  [{i}/{len(questions)}] {qid} ({tier}): {question_text[:60]}…")

        try:
            result = rag_answer(
                question_text,
                collection=collection,
                use_reranker=use_reranker,
            )
            contexts = [c.text_preview for c in result.cited_chunks if c.text_preview]
            # Pad with top retrieved texts if no citations were parsed
            if not contexts:
                contexts = ["[no context retrieved]"]

            records.append(
                {
                    "id": qid,
                    "tier": tier,
                    "question": question_text,
                    "answer": result.answer,
                    "gold_answer": gold,
                    "contexts": contexts,
                    "rewritten_query": result.rewritten_query,
                    "timings_ms": result.timings_ms.model_dump(),
                    "est_cost_usd": result.est_cost_usd,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )
        except Exception as e:
            logger.error("pipeline error on %s: %s", qid, e)
            errors += 1
            records.append(
                {
                    "id": qid,
                    "tier": tier,
                    "question": question_text,
                    "answer": f"[ERROR: {e}]",
                    "gold_answer": gold,
                    "contexts": ["[error]"],
                    "rewritten_query": question_text,
                    "timings_ms": {},
                    "est_cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

    if errors:
        console.print(f"[yellow]  {errors} questions failed pipeline execution[/yellow]")

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    valid_records = [r for r in records if not r["answer"].startswith("[ERROR")]
    if not valid_records:
        console.print("[red]No valid records to evaluate.[/red]")
        return records, {}

    console.print(f"\n[bold]Running RAGAS evaluation on {len(valid_records)} records…[/bold]")

    try:
        from ragas import evaluate

        dataset = _build_ragas_dataset(valid_records)
        metrics = _get_metrics()
        ragas_result = evaluate(dataset, metrics=metrics)

        # Extract scores dict — handles both old (df-based) and new (dict-based) APIs
        if hasattr(ragas_result, "to_pandas"):
            df = ragas_result.to_pandas()
            global_scores = {col: float(df[col].mean()) for col in df.columns if col in METRICS_ORDER}
        else:
            global_scores = {k: float(v) for k, v in ragas_result.items() if k in METRICS_ORDER}

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        global_scores = {}

    # ── Per-tier breakdown ────────────────────────────────────────────────────
    tier_scores: dict[str, dict] = {}
    for tier in TIER_ORDER:
        tier_records = [r for r in valid_records if r["tier"] == tier]
        if not tier_records:
            continue
        try:
            from ragas import evaluate

            tier_ds = _build_ragas_dataset(tier_records)
            tier_result = evaluate(tier_ds, metrics=_get_metrics())
            if hasattr(tier_result, "to_pandas"):
                df = tier_result.to_pandas()
                tier_scores[tier] = {col: float(df[col].mean()) for col in df.columns if col in METRICS_ORDER}
            else:
                tier_scores[tier] = {k: float(v) for k, v in tier_result.items() if k in METRICS_ORDER}
        except Exception as e:
            logger.warning("per-tier RAGAS for '%s' failed: %s", tier, e)

    results = {
        "global": global_scores,
        "per_tier": tier_scores,
        "n_questions": len(valid_records),
        "n_errors": errors,
        "avg_cost_usd": sum(r["est_cost_usd"] for r in valid_records) / max(len(valid_records), 1),
        "avg_latency_ms": sum(
            sum(r["timings_ms"].values()) for r in valid_records if r["timings_ms"]
        ) / max(len(valid_records), 1),
    }
    results.update(global_scores)  # flatten faithfulness etc to top level for CI gate
    return records, results


def _print_results_table(results: dict, label: str = "") -> None:
    title = f"RAGAS Results{' — ' + label if label else ''}"
    tbl = Table(title=title, show_lines=True)
    tbl.add_column("Tier", style="bold")
    for m in METRICS_ORDER:
        tbl.add_column(m.replace("_", " ").title(), justify="right")
    tbl.add_column("N", justify="right")

    # Global row
    g = results.get("global", {})
    tbl.add_row(
        "[bold]Overall[/bold]",
        *[f"{g.get(m, 0):.4f}" for m in METRICS_ORDER],
        str(results.get("n_questions", "?")),
    )

    # Per-tier rows
    for tier in TIER_ORDER:
        t = results.get("per_tier", {}).get(tier, {})
        if t:
            tbl.add_row(
                tier.capitalize(),
                *[f"{t.get(m, 0):.4f}" for m in METRICS_ORDER],
                "",
            )

    console.print(tbl)
    console.print(
        f"Avg cost/query: ${results.get('avg_cost_usd', 0):.5f}  "
        f"Avg latency: {results.get('avg_latency_ms', 0):.0f}ms\n"
    )


def _write_markdown_table(results: dict, path: Path, label: str = "") -> None:
    lines = [f"## RAGAS Results{' — ' + label if label else ''}\n"]
    header = "| Tier | " + " | ".join(m.replace("_", " ").title() for m in METRICS_ORDER) + " | N |"
    sep = "|------|" + "---------|" * len(METRICS_ORDER) + "-----|"
    lines += [header, sep]

    g = results.get("global", {})
    lines.append(
        f"| **Overall** | "
        + " | ".join(f"**{g.get(m, 0):.4f}**" for m in METRICS_ORDER)
        + f" | **{results.get('n_questions', '?')}** |"
    )
    for tier in TIER_ORDER:
        t = results.get("per_tier", {}).get(tier, {})
        if t:
            lines.append(
                f"| {tier.capitalize()} | "
                + " | ".join(f"{t.get(m, 0):.4f}" for m in METRICS_ORDER)
                + " | |"
            )

    lines.append(
        f"\n_Avg cost/query: ${results.get('avg_cost_usd', 0):.5f} | "
        f"Avg latency: {results.get('avg_latency_ms', 0):.0f}ms_\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the benchmark")
    parser.add_argument("--smoke", action="store_true", help="10-question CI subset")
    parser.add_argument("--tier", choices=TIER_ORDER, default=None, help="run a single tier only")
    parser.add_argument("--no-reranker", action="store_true", help="skip cross-encoder reranking")
    parser.add_argument("--collection", default=None, help="Qdrant collection name override")
    parser.add_argument("--output-json", default=None, help="write results JSON to this path")
    parser.add_argument("--output-md", default=None, help="write markdown table to this path")
    args = parser.parse_args()

    questions = _load_benchmark(tier=args.tier, smoke=args.smoke)
    if not questions:
        console.print("[red]No questions matched the filter.[/red]")
        sys.exit(1)

    console.print(f"[bold]Benchmark: {len(questions)} questions[/bold]  (smoke={args.smoke}, tier={args.tier})")

    records, results = run_eval(
        questions,
        use_reranker=not args.no_reranker,
        collection=args.collection,
    )

    _print_results_table(results, label=args.tier or ("smoke" if args.smoke else "full"))

    # Write JSON
    out_json = Path(args.output_json) if args.output_json else None
    if out_json:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {"smoke": args.smoke, "tier": args.tier, "reranker": not args.no_reranker},
            "results": results,
            "records": records,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"Results written → {out_json}")

    # Write markdown
    out_md = Path(args.output_md) if args.output_md else None
    if out_md:
        _write_markdown_table(results, out_md, label=args.tier or ("smoke" if args.smoke else "full"))
        console.print(f"Markdown table written → {out_md}")

    # CI gate: fail if faithfulness below threshold
    faithfulness = results.get("faithfulness", results.get("global", {}).get("faithfulness", None))
    if faithfulness is not None:
        threshold = settings.faithfulness_threshold
        if faithfulness < threshold:
            console.print(
                f"[red bold]EVAL GATE FAILED: faithfulness {faithfulness:.4f} < {threshold:.2f}[/red bold]"
            )
            sys.exit(1)
        else:
            console.print(
                f"[green bold]EVAL GATE PASSED: faithfulness {faithfulness:.4f} >= {threshold:.2f}[/green bold]"
            )
