"""RAG answer generation with citations, timings, and cost estimation.

Returns a RagAnswer Pydantic model with:
  - answer (str): the generated text with inline [doc_id:chunk_id] citations
  - cited_chunks (list[CitedChunk]): resolved source chunks
  - timings_ms (TimingBreakdown): per-step latency
  - est_cost_usd (float): token-based cost estimate
  - faithfulness_score (float | None): inline RAGAS score if requested
"""

from __future__ import annotations

from pydantic import BaseModel


class CitedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    url: str
    text_preview: str  # first 300 chars
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class TimingBreakdown(BaseModel):
    rewrite_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.rewrite_ms + self.embed_ms + self.retrieve_ms + self.rerank_ms + self.generate_ms


class RagAnswer(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    cited_chunks: list[CitedChunk]
    timings_ms: TimingBreakdown
    est_cost_usd: float
    faithfulness_score: float | None = None


# TODO Phase 3: implement
#   async def answer(question: str, config: RetrievalConfig) -> RagAnswer:
#       - rewrite query
#       - embed query
#       - hybrid retrieve + rerank
#       - call generator with SYSTEM_PROMPT + USER_TEMPLATE
#       - parse [doc_id:chunk_id] markers from response
#       - compute cost from token counts
#       - log trace to Langfuse
