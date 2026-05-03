"""Debug retrieval: pretty-print the full trace for a query.

Shows at each stage:
  1. Original query
  2. Rewritten query
  3. Top-K vector hits (with scores)
  4. Top-K BM25 hits (with scores)
  5. RRF-merged ranking (with fused scores)
  6. Post-rerank ranking (with cross-encoder scores)

Used by:
  - Streamlit "Debug retrieval" toggle (Phase 5)
  - CLI flag:  uv run python -m erag.retrieve.debug "your question here"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from erag.config import PROCESSED_DIR, settings
from erag.index.embed import embed_query
from erag.retrieve.bm25 import bm25_search
from erag.retrieve.hybrid import RankedResult, reciprocal_rank_fusion
from erag.retrieve.rerank import rerank
from erag.retrieve.rewrite import rewrite_query
from erag.retrieve.vector import vector_search

logger = logging.getLogger(__name__)
console = Console()

BM25_PATH = PROCESSED_DIR / "bm25_{collection}.pkl"


@dataclass
class RetrievalTrace:
    original_query: str
    rewritten_query: str
    vector_hits: list[RankedResult]
    bm25_hits: list[RankedResult]
    rrf_merged: list[RankedResult]
    reranked: list[RankedResult]
    timings_ms: dict[str, float]


def retrieve_with_trace(
    question: str,
    collection: str | None = None,
    top_k_vector: int | None = None,
    top_k_bm25: int | None = None,
    top_k_rerank: int | None = None,
    use_reranker: bool = True,
) -> RetrievalTrace:
    """Run the full retrieval pipeline and return a structured trace."""
    from erag.index.qdrant_client import get_client

    collection = collection or settings.qdrant_collection
    top_k_vector = top_k_vector or settings.top_k_vector
    top_k_bm25 = top_k_bm25 or settings.top_k_bm25
    top_k_rerank = top_k_rerank or settings.top_k_rerank
    bm25_path = Path(str(BM25_PATH).format(collection=collection))

    timings: dict[str, float] = {}

    # 1. Rewrite
    t0 = time.perf_counter()
    rewritten = rewrite_query(question)
    timings["rewrite_ms"] = (time.perf_counter() - t0) * 1000

    # 2. Embed
    t0 = time.perf_counter()
    query_vec = embed_query(rewritten, settings.embedding_model)
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    # 3. Vector search
    t0 = time.perf_counter()
    client = get_client(settings.qdrant_url, settings.qdrant_api_key)
    vector_hits = vector_search(query_vec.tolist(), top_k_vector, client, collection)
    timings["vector_ms"] = (time.perf_counter() - t0) * 1000

    # 4. BM25 search
    bm25_hits: list[RankedResult] = []
    if bm25_path.exists():
        t0 = time.perf_counter()
        bm25_hits = bm25_search(rewritten, top_k_bm25, bm25_path)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
    else:
        logger.warning("BM25 index not found at %s — skipping BM25", bm25_path)
        timings["bm25_ms"] = 0.0

    # 5. RRF merge
    t0 = time.perf_counter()
    rrf_merged = reciprocal_rank_fusion(vector_hits, bm25_hits, k=settings.rrf_k)
    timings["rrf_ms"] = (time.perf_counter() - t0) * 1000

    # 6. Rerank
    reranked: list[RankedResult] = []
    if use_reranker:
        t0 = time.perf_counter()
        reranked = rerank(rewritten, rrf_merged, top_k_rerank, settings.reranker_model)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    else:
        reranked = rrf_merged[:top_k_rerank]
        timings["rerank_ms"] = 0.0

    return RetrievalTrace(
        original_query=question,
        rewritten_query=rewritten,
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
        rrf_merged=rrf_merged,
        reranked=reranked,
        timings_ms=timings,
    )


def print_trace(trace: RetrievalTrace) -> None:
    """Pretty-print a RetrievalTrace to the terminal using Rich."""
    console.rule("[bold cyan]Retrieval Debug Trace")

    console.print(f"\n[bold]Original query:[/bold]  {trace.original_query}")
    console.print(f"[bold]Rewritten query:[/bold] [green]{trace.rewritten_query}[/green]")

    t = trace.timings_ms
    console.print(
        f"\n[dim]Timings: rewrite={t.get('rewrite_ms',0):.0f}ms  "
        f"embed={t.get('embed_ms',0):.0f}ms  "
        f"vector={t.get('vector_ms',0):.0f}ms  "
        f"bm25={t.get('bm25_ms',0):.0f}ms  "
        f"rerank={t.get('rerank_ms',0):.0f}ms[/dim]"
    )

    def _result_table(title: str, results: list[RankedResult], score_label: str = "score") -> Table:
        tbl = Table(title=title, show_lines=True)
        tbl.add_column("#", style="dim", width=3)
        tbl.add_column(score_label, width=8)
        tbl.add_column("chunk_id", width=35)
        tbl.add_column("title / preview", overflow="fold")
        for rank, r in enumerate(results, 1):
            preview = r.metadata.get("text", "")[:80].replace("\n", " ")
            title_str = r.metadata.get("title", "")[:40]
            tbl.add_row(str(rank), f"{r.score:.4f}", r.chunk_id, f"{title_str}\n[dim]{preview}…[/dim]")
        return tbl

    console.print()
    console.print(_result_table(f"Vector hits (top {len(trace.vector_hits)})", trace.vector_hits, "cosine"))
    console.print()
    console.print(_result_table(f"BM25 hits (top {len(trace.bm25_hits)})", trace.bm25_hits, "bm25"))
    console.print()
    console.print(_result_table(f"RRF merged (top {len(trace.rrf_merged)})", trace.rrf_merged, "rrf"))
    console.print()
    console.print(_result_table(f"Post-rerank (top {len(trace.reranked)})", trace.reranked, "ce_score"))
    console.rule()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Debug retrieval for a query")
    parser.add_argument("question", help="The question to retrieve for")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--no-rerank", action="store_true", help="skip cross-encoder reranking")
    parser.add_argument("--json", dest="as_json", action="store_true", help="output JSON instead of rich tables")
    args = parser.parse_args()

    trace = retrieve_with_trace(
        args.question,
        collection=args.collection,
        use_reranker=not args.no_rerank,
    )

    if args.as_json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(trace), indent=2, default=str))
    else:
        print_trace(trace)
