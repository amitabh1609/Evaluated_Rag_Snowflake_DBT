"""Run the 3-config ablation and write docs/ablation_results.md.

Configs:
  A — fixed-512 chunks, vector-only retrieval, no reranker  (baseline)
  B — fixed-512 chunks, vector-only retrieval, BGE reranker
  C — semantic chunks,  hybrid (vector + BM25 + RRF), BGE reranker  ← primary

For each config:
  1. Build a dedicated Qdrant collection (erag_ablation_<config>)
  2. Run ragas_runner.run_eval() on the full 50-question benchmark
  3. Collect metrics and latency
  4. Write docs/ablation_results.md with the comparison table

Total runtime: ~60–90 minutes (dominated by embedding + RAGAS API calls).

Usage:
    uv run python -m erag.eval.ablation
    uv run python -m erag.eval.ablation --smoke    # 10-question subset (faster)
    uv run python -m erag.eval.ablation --config C # single config
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from erag.config import PROCESSED_DIR, settings
from erag.eval.ragas_runner import METRICS_ORDER, TIER_ORDER, _load_benchmark, _write_markdown_table, run_eval

logger = logging.getLogger(__name__)
console = Console()

RESULTS_PATH = Path(__file__).resolve().parents[4] / "docs" / "ablation_results.md"


@dataclass
class AblationConfig:
    label: str          # "A", "B", "C"
    description: str
    chunker: str        # "fixed" | "semantic"
    use_bm25: bool
    use_reranker: bool
    collection: str     # dedicated Qdrant collection name


CONFIGS = [
    AblationConfig(
        label="A",
        description="fixed-512, vector-only, no reranker (baseline)",
        chunker="fixed",
        use_bm25=False,
        use_reranker=False,
        collection="erag_ablation_a",
    ),
    AblationConfig(
        label="B",
        description="fixed-512, vector-only, BGE reranker",
        chunker="fixed",
        use_bm25=False,
        use_reranker=True,
        collection="erag_ablation_b",
    ),
    AblationConfig(
        label="C",
        description="semantic chunks, hybrid+RRF, BGE reranker (primary)",
        chunker="semantic",
        use_bm25=True,
        use_reranker=True,
        collection="erag_ablation_c",
    ),
]


def _build_config_index(cfg: AblationConfig) -> None:
    """Build Qdrant + BM25 index for a given ablation config."""
    from erag.index.build_index import build

    console.print(f"\n[bold]Building index for Config {cfg.label}: {cfg.description}[/bold]")
    build(
        chunker=cfg.chunker,
        collection=cfg.collection,
        smoke=False,
        reset=True,  # always rebuild for ablation reproducibility
    )


def _patch_settings_for_config(cfg: AblationConfig) -> None:
    """Temporarily patch settings to disable BM25 when use_bm25=False.

    We do this by moving the BM25 index file to a temp location so
    bm25_search() naturally skips it (file not found path).
    """
    bm25_path = PROCESSED_DIR / f"bm25_{cfg.collection}.pkl"
    bm25_temp = PROCESSED_DIR / f"bm25_{cfg.collection}.pkl.disabled"

    if not cfg.use_bm25 and bm25_path.exists():
        bm25_path.rename(bm25_temp)
        logger.info("BM25 disabled for config %s (renamed index)", cfg.label)
    elif cfg.use_bm25 and bm25_temp.exists():
        bm25_temp.rename(bm25_path)


def _restore_bm25(cfg: AblationConfig) -> None:
    bm25_temp = PROCESSED_DIR / f"bm25_{cfg.collection}.pkl.disabled"
    bm25_path = PROCESSED_DIR / f"bm25_{cfg.collection}.pkl"
    if bm25_temp.exists():
        bm25_temp.rename(bm25_path)


def run_ablation(smoke: bool = False, config_filter: str | None = None) -> None:
    questions = _load_benchmark(smoke=smoke)
    configs = [c for c in CONFIGS if config_filter is None or c.label == config_filter]

    all_results: dict[str, dict] = {}

    for cfg in configs:
        console.rule(f"[bold cyan]Config {cfg.label} — {cfg.description}")

        _build_config_index(cfg)
        _patch_settings_for_config(cfg)

        try:
            _, results = run_eval(
                questions,
                use_reranker=cfg.use_reranker,
                collection=cfg.collection,
            )
        finally:
            _restore_bm25(cfg)

        all_results[cfg.label] = {"config": cfg, "results": results}
        console.print(f"Config {cfg.label} done: faithfulness={results.get('faithfulness', 'N/A')}")

    _write_ablation_md(all_results, smoke=smoke)
    _write_ablation_json(all_results, smoke=smoke)
    console.print(f"\n[green bold]Ablation complete → {RESULTS_PATH}[/green bold]")


def _write_ablation_md(all_results: dict, smoke: bool = False) -> None:
    lines = [
        "# Ablation Results",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Mode: {'smoke (10 questions)' if smoke else 'full benchmark (50 questions)'}_",
        "",
        "## 3-Config Comparison",
        "",
        "| Config | Chunking | Retrieval | Reranker | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Avg Latency | Avg $/query |",
        "|--------|----------|-----------|----------|-------------|-------------|---------------|------------|-------------|-------------|",
    ]

    config_meta = {
        "A": ("fixed-512", "vector-only", "none"),
        "B": ("fixed-512", "vector-only", "BGE-rerank-L"),
        "C": ("semantic", "hybrid+RRF", "BGE-rerank-L"),
    }

    for label, meta in config_meta.items():
        chunking, retrieval, reranker = meta
        if label not in all_results:
            lines.append(f"| {label} | {chunking} | {retrieval} | {reranker} | — | — | — | — | — | — |")
            continue
        r = all_results[label]["results"]
        g = r.get("global", {})
        f = g.get("faithfulness", r.get("faithfulness", 0))
        ar = g.get("answer_relevancy", r.get("answer_relevancy", 0))
        cp = g.get("context_precision", r.get("context_precision", 0))
        cr = g.get("context_recall", r.get("context_recall", 0))
        lat = r.get("avg_latency_ms", 0)
        cost = r.get("avg_cost_usd", 0)
        lines.append(
            f"| **{label}** | {chunking} | {retrieval} | {reranker} | "
            f"**{f:.4f}** | {ar:.4f} | {cp:.4f} | {cr:.4f} | {lat:.0f}ms | ${cost:.5f} |"
        )

    # Per-tier breakdown for Config C
    if "C" in all_results:
        c_results = all_results["C"]["results"]
        lines += [
            "",
            "## Config C — Per-Tier Breakdown",
            "",
            "| Tier | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | N |",
            "|------|-------------|-------------|---------------|------------|---|",
        ]
        tier_counts = {"factual": 15, "conceptual": 15, "architectural": 10, "adversarial": 10}
        if smoke:
            tier_counts = {t: "~" for t in TIER_ORDER}

        for tier in TIER_ORDER:
            t = c_results.get("per_tier", {}).get(tier, {})
            n = tier_counts.get(tier, "?")
            if t:
                lines.append(
                    f"| {tier.capitalize()} | "
                    f"{t.get('faithfulness', 0):.4f} | "
                    f"{t.get('answer_relevancy', 0):.4f} | "
                    f"{t.get('context_precision', 0):.4f} | "
                    f"{t.get('context_recall', 0):.4f} | {n} |"
                )
            else:
                lines.append(f"| {tier.capitalize()} | — | — | — | — | {n} |")

        g = c_results.get("global", {})
        lines.append(
            f"| **Overall** | "
            f"**{g.get('faithfulness', 0):.4f}** | "
            f"**{g.get('answer_relevancy', 0):.4f}** | "
            f"**{g.get('context_precision', 0):.4f}** | "
            f"**{g.get('context_recall', 0):.4f}** | **50** |"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_json(all_results: dict, smoke: bool = False) -> None:
    json_path = RESULTS_PATH.with_suffix(".json")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "configs": {
            label: {
                "description": v["config"].description,
                "chunker": v["config"].chunker,
                "use_bm25": v["config"].use_bm25,
                "use_reranker": v["config"].use_reranker,
                "results": v["results"],
            }
            for label, v in all_results.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run 3-config ablation")
    parser.add_argument("--smoke", action="store_true", help="10-question subset")
    parser.add_argument("--config", choices=["A", "B", "C"], default=None, help="run a single config")
    args = parser.parse_args()

    run_ablation(smoke=args.smoke, config_filter=args.config)
