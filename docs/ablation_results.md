# Ablation Results

> Populated in Phase 4 after running `make ablation`.
> All numbers are reproducible by running `make ablation` on a clean checkout.

## 3-Config Comparison (full 50-question benchmark)

| Config | Chunking | Retrieval | Reranker | Faithfulness | Answer Rel. | Ctx Precision | P50 latency | $ / query |
|--------|----------|-----------|----------|-------------|-------------|---------------|-------------|-----------|
| A (baseline) | fixed-512 | vector-only | none | TBD | TBD | TBD | TBDms | $TBD |
| B | fixed-512 | vector-only | BGE-rerank-L | TBD | TBD | TBD | TBDms | $TBD |
| C (primary) | semantic | hybrid+RRF | BGE-rerank-L | TBD | TBD | TBD | TBDms | $TBD |

## Config C — Per-Tier Breakdown

| Tier | Faithfulness | Answer Rel. | Ctx Precision | N |
|------|-------------|-------------|---------------|---|
| Factual | TBD | TBD | TBD | 15 |
| Conceptual | TBD | TBD | TBD | 15 |
| Architectural | TBD | TBD | TBD | 10 |
| Adversarial | TBD | TBD | TBD | 10 |
| **Overall** | **TBD** | **TBD** | **TBD** | **50** |
