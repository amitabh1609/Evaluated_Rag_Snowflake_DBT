# Evaluated RAG System for Snowflake & dbt Documentation

<!-- CI badge — added once the Actions workflow is green -->
<!-- ![Eval Gate](https://github.com/amitabh1609/evaluated_rag_snowflake_dbt/actions/workflows/eval.yml/badge.svg) -->

> **Headline metric:** _To be filled after Phase 4 ablation — e.g., "Faithfulness 0.XX,
> Precision@3 0.XX, P95 latency X.Xs, $0.00X / query on a 50-question hand-curated
> benchmark across 4 difficulty tiers."_

---

## What this is

<!-- 3–4 sentences: hybrid retrieval, hand-evaluated benchmark, honest failure analysis -->
_TODO — write after Phase 4._

---

## Live demo

<!-- Screenshot of Streamlit UI + Loom link -->
_TODO — add after Phase 5._

---

## Architecture

<!-- Insert docs/architecture.png here -->
_TODO — diagram after Phase 2._

```
crawl → chunk → embed → hybrid retrieve (vector + BM25 + RRF)
      → rerank → generate → evaluate
```

---

## Why this corpus

<!-- 1 paragraph: Snowflake architecture + warehouse + table-format docs,
     dbt official best-practices, dbt Discourse top-200 threads -->
_TODO_

---

## Search merge logic

<!-- RRF formula, k=60, why it beats simple concatenation -->
_TODO_

---

## Evaluation methodology

<!-- 4-tier benchmark, why hand-curated, RAGAS metrics used -->
_TODO_

---

## Ablation results

<!-- 3-config table linking to docs/ablation_results.md -->
_TODO — see [docs/ablation_results.md](docs/ablation_results.md)_

---

## Latency and cost

<!-- Langfuse screenshot + per-step breakdown numbers -->
_TODO_

---

## What didn't work

<!-- ≥5 honest entries — the section recruiters read most carefully -->
_TODO — curated from [WHAT_I_GOT_WRONG.md](WHAT_I_GOT_WRONG.md) at the end._

---

## What I'd do differently

_TODO_

---

## Run it locally

```bash
git clone https://github.com/amitabh1609/evaluated_rag_snowflake_dbt
cd evaluated_rag_snowflake_dbt
cp .env.example .env          # fill in API keys
make install                  # uv sync
make up                       # Qdrant + Langfuse
make crawl                    # ~1–2 hours, polite crawl
make index                    # embed + upsert
make ui                       # Streamlit on :8501
make eval                     # full RAGAS evaluation
```

---

## Decisions log

See [DECISIONS.md](DECISIONS.md) for every non-obvious architectural choice.
