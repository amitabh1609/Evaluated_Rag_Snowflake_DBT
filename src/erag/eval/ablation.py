"""Run the 3-config ablation and write docs/ablation_results.md.

Config A: fixed-512 chunks, vector-only retrieval, no reranker (baseline)
Config B: fixed-512 chunks, vector-only retrieval, BGE reranker
Config C: semantic chunks, hybrid (vector + BM25 + RRF), BGE reranker

For each config: builds a fresh Qdrant collection, runs ragas_runner, records metrics.
Total runtime ~60–90 minutes (dominated by embedding and RAGAS API calls).

Usage:
    python -m erag.eval.ablation
"""

# TODO Phase 4: implement
#   - Define RetrievalConfig dataclass with chunker / retriever / reranker fields
#   - For each config: build_index → ragas_runner → collect row
#   - Write markdown table to docs/ablation_results.md
