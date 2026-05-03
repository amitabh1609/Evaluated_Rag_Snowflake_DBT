"""RAG answer generation with citations, per-step timings, and cost estimation.

Pipeline per query:
  rewrite → embed → vector+BM25 retrieve → RRF → rerank → generate → parse citations

Returns a RagAnswer Pydantic model. All steps are traced to Langfuse.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Per-model token pricing (USD per 1M tokens, as of 2025-05) ───────────────
_COST_TABLE: dict[str, dict[str, float]] = {
    "claude-opus-4-7":            {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":          {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input":  0.25, "output":  1.25},
    "gpt-4o":                     {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":                {"input":  0.15, "output":  0.60},
}
_DEFAULT_COST = {"input": 3.00, "output": 15.00}

_CITATION_RE = re.compile(r"\[([^:\]\s]+):([^\]\s]+)\]")


# ── Pydantic response models ──────────────────────────────────────────────────

class CitedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    title: str = ""
    url: str = ""
    source: str = ""
    text_preview: str = ""   # first 300 chars
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class TimingBreakdown(BaseModel):
    rewrite_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.rewrite_ms
            + self.embed_ms
            + self.retrieve_ms
            + self.rerank_ms
            + self.generate_ms
        )


class RagAnswer(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    cited_chunks: list[CitedChunk]
    timings_ms: TimingBreakdown
    est_cost_usd: float
    faithfulness_score: float | None = None
    # token counts for UI display
    input_tokens: int = 0
    output_tokens: int = 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _COST_TABLE.get(model, _DEFAULT_COST)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _parse_citations(answer_text: str, chunk_lookup: dict[str, dict]) -> list[CitedChunk]:
    """Extract [doc_id:chunk_index] markers and resolve to CitedChunk objects.

    The LLM writes citations as [doc_id:chunk_index] (e.g. [sf_wh:3]).
    Internally chunk_ids are stored as "doc_id:index" (e.g. "sf_wh:3"), so
    we reconstruct the full chunk_id = f"{doc_id}:{chunk_suffix}" before lookup.
    """
    seen: dict[str, CitedChunk] = {}
    for match in _CITATION_RE.finditer(answer_text):
        doc_id, chunk_suffix = match.group(1), match.group(2)
        full_chunk_id = f"{doc_id}:{chunk_suffix}"   # matches stored chunk_id
        if full_chunk_id in seen:
            continue
        payload = chunk_lookup.get(full_chunk_id, {})
        seen[full_chunk_id] = CitedChunk(
            doc_id=payload.get("doc_id", doc_id),
            chunk_id=full_chunk_id,
            title=payload.get("title", ""),
            url=payload.get("url", ""),
            source=payload.get("source", ""),
            text_preview=payload.get("text", "")[:300],
            vector_score=payload.get("vector_score"),
            bm25_score=payload.get("bm25_score"),
            rerank_score=payload.get("rerank_score"),
            rrf_score=payload.get("rrf_score"),
        )
    return list(seen.values())


def _call_anthropic(
    model: str,
    system: str,
    user_message: str,
    api_key: str,
) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    text = msg.content[0].text
    return text, msg.usage.input_tokens, msg.usage.output_tokens


def _call_openai(
    model: str,
    system: str,
    user_message: str,
    api_key: str,
) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    choice = resp.choices[0]
    usage = resp.usage
    return choice.message.content, usage.prompt_tokens, usage.completion_tokens


# ── Public API ────────────────────────────────────────────────────────────────

def answer(
    question: str,
    collection: str | None = None,
    use_reranker: bool = True,
    compute_faithfulness: bool = False,
) -> RagAnswer:
    """Run the full RAG pipeline and return a structured answer with citations.

    Args:
        question: Raw user question.
        collection: Qdrant collection name (defaults to settings.qdrant_collection).
        use_reranker: Whether to run the BGE cross-encoder reranking step.
        compute_faithfulness: If True, compute an inline RAGAS faithfulness score.

    Returns:
        RagAnswer with answer text, cited chunks, timings, and cost.
    """
    from erag.config import PROCESSED_DIR, settings
    from erag.generate.prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
    from erag.index.embed import embed_query
    from erag.index.qdrant_client import get_client
    from erag.observe import langfuse_client as lf
    from erag.retrieve.bm25 import bm25_search
    from erag.retrieve.hybrid import reciprocal_rank_fusion
    from erag.retrieve.rerank import rerank
    from erag.retrieve.rewrite import rewrite_query
    from erag.retrieve.vector import vector_search

    collection = collection or settings.qdrant_collection
    bm25_path = PROCESSED_DIR / f"bm25_{collection}.pkl"

    timings = TimingBreakdown()
    trace = lf.start_trace(
        name="rag_query",
        input={"question": question, "collection": collection},
    )

    # ── 1. Rewrite ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with lf.span(trace, "rewrite", input={"question": question}) as sp:
        rewritten = rewrite_query(question, settings.rewriter_model)
        sp.end(output={"rewritten": rewritten})
    timings.rewrite_ms = (time.perf_counter() - t0) * 1000

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with lf.span(trace, "embed", input={"query": rewritten}) as sp:
        query_vec = embed_query(rewritten, settings.embedding_model)
        sp.end(output={"dim": len(query_vec)})
    timings.embed_ms = (time.perf_counter() - t0) * 1000

    # ── 3. Retrieve (vector + BM25 + RRF) ────────────────────────────────────
    t0 = time.perf_counter()
    with lf.span(trace, "retrieve", input={"rewritten_query": rewritten}) as sp:
        qdrant = get_client(settings.qdrant_url, settings.qdrant_api_key)
        vector_hits = vector_search(query_vec.tolist(), settings.top_k_vector, qdrant, collection)

        bm25_hits = []
        if bm25_path.exists():
            bm25_hits = bm25_search(rewritten, settings.top_k_bm25, bm25_path)

        merged = reciprocal_rank_fusion(vector_hits, bm25_hits, k=settings.rrf_k)
        sp.end(output={"vector_hits": len(vector_hits), "bm25_hits": len(bm25_hits), "merged": len(merged)})
    timings.retrieve_ms = (time.perf_counter() - t0) * 1000

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with lf.span(trace, "rerank", input={"candidates": len(merged)}) as sp:
        if use_reranker and merged:
            final_results = rerank(rewritten, merged, settings.top_k_rerank, settings.reranker_model)
        else:
            final_results = merged[: settings.top_k_rerank]
        sp.end(output={"results": len(final_results)})
    timings.rerank_ms = (time.perf_counter() - t0) * 1000

    # Build lookup keyed by chunk_id (format: "doc_id:index") for citation resolution
    chunk_lookup: dict[str, dict] = {}
    for r in final_results:
        chunk_lookup[r.chunk_id] = {**r.metadata, "rerank_score": r.score}

    # ── 5. Generate ───────────────────────────────────────────────────────────
    context_chunks = [r.metadata for r in final_results]
    context_str = format_context(context_chunks)
    user_msg = USER_TEMPLATE.format(context=context_str, question=question)

    t0 = time.perf_counter()
    with lf.span(trace, "generate", input={"model": settings.generator_model}) as sp:
        if settings.generator_model.startswith("claude"):
            raw_answer, in_tok, out_tok = _call_anthropic(
                settings.generator_model, SYSTEM_PROMPT, user_msg, settings.anthropic_api_key
            )
        else:
            raw_answer, in_tok, out_tok = _call_openai(
                settings.generator_model, SYSTEM_PROMPT, user_msg, settings.openai_api_key
            )
        sp.end(output={"answer_len": len(raw_answer), "input_tokens": in_tok, "output_tokens": out_tok})
    timings.generate_ms = (time.perf_counter() - t0) * 1000

    cost = _cost(settings.generator_model, in_tok, out_tok)
    cited = _parse_citations(raw_answer, chunk_lookup)

    # ── 6. Optional inline faithfulness ──────────────────────────────────────
    faithfulness: float | None = None
    if compute_faithfulness and cited:
        try:
            faithfulness = _inline_faithfulness(question, raw_answer, [r.metadata.get("text", "") for r in final_results])
        except Exception as e:
            logger.warning("inline faithfulness failed: %s", e)

    trace.update(
        output={"answer": raw_answer[:200], "cost_usd": cost, "faithfulness": faithfulness}
    )
    lf.flush()

    return RagAnswer(
        question=question,
        rewritten_query=rewritten,
        answer=raw_answer,
        cited_chunks=cited,
        timings_ms=timings,
        est_cost_usd=cost,
        faithfulness_score=faithfulness,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _inline_faithfulness(question: str, answer_text: str, contexts: list[str]) -> float:
    """Lightweight single-question faithfulness via RAGAS (used in UI).

    Runs a one-question RAGAS evaluation and returns the faithfulness score [0, 1].
    Expensive (~1 LLM call) — only called when explicitly requested.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer_text],
        "contexts": [contexts],
    })
    result = evaluate(ds, metrics=[faithfulness])
    return float(result["faithfulness"])
