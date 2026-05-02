# Project Brief — Evaluated RAG System for Snowflake & dbt Documentation

> This file is kept in the repo so every contributor (and future Claude Code session)
> has the full context in one place. Do not delete or edit this file without updating
> the corresponding section in README.md or DECISIONS.md.

---

## 1. Who I am and what we're building

I am Amitabh Choudhury, a Data Engineer with ~3 years at Caterpillar (Snowflake, dbt, Airflow,
Cortex prototypes) transitioning into mid-level AI / GenAI Engineer and senior-leaning Data
Engineer roles in Bengaluru, Singapore, Dubai, and Switzerland.

This repo is **Project 2 of a 3-project portfolio**. The recruiter-facing claim it must defend is:

> "I built an evaluated RAG system over Snowflake + dbt documentation, with hybrid retrieval,
> a 4-tier hand-curated benchmark, a CI gate on faithfulness, and an honest failure-analysis
> section."

The README is the deliverable. The code is the backing evidence.

Project name on GitHub: `evaluated-rag-snowflake-dbt`
Project title in README: **Evaluated RAG System for Snowflake & dbt Documentation**
(Do not call this "Production RAG." "Evaluated" is the credible word.)

---

## 2. Target outcome — what "done" looks like

A reviewer landing on the repo should, within 90 seconds, be able to see:

- A one-line **headline metric**: "Faithfulness 0.XX, Precision@3 0.XX, P95 latency X.Xs,
  $0.00X / query, on a 50-question hand-curated benchmark across 4 difficulty tiers."
- An **architecture diagram** (boxes and arrows: crawl → chunk → embed → hybrid retrieve →
  rerank → generate → evaluate).
- A `docker-compose up` **quickstart** that brings up Qdrant + Langfuse + the Streamlit UI locally.
- A linked [DECISIONS.md](DECISIONS.md) for every non-obvious choice.
- A **"What didn't work"** section with at least 5 honest entries.
- A **green CI badge** from a GitHub Actions workflow that runs RAGAS on every PR and fails
  the build if faithfulness drops below 0.80.

---

## 3. Non-negotiable differentiators

### Retrieval quality
- **Hybrid search**: Vector (BGE-large-en-v1.5) + BM25 merged with RRF.
- **LLM-assisted query rewriting**: GPT-4o-mini or Claude Haiku rewrites vague questions.
- **Retrieval debugging layer**: Streamlit side panel + `--debug-retrieval` CLI flag.

### Evaluation methodology
- **4-tier hand-curated benchmark** (50 questions in `evals/benchmark.yaml`):
  - Factual lookup (15)
  - Conceptual (15)
  - Architectural (10)
  - Adversarial (10)
- **One committed ablation**, three configs, real numbers:
  - Config A: fixed 512-token, vector-only, no reranker (baseline)
  - Config B: fixed 512-token, vector-only, BGE reranker
  - Config C: semantic chunking, hybrid (vector + BM25 + RRF), BGE reranker
- **Failure analysis**: 5+ benchmark failures with diagnosis and remediation attempts.

### CI/CD and production signals
- **GitHub Actions eval gate**: fails build if faithfulness < 0.80.
- **Latency + cost breakdown** in the UI.
- **Trust signals**: cited source chunks + per-answer faithfulness score.

### Corpus and framing
- Snowflake architecture + warehouse + table-format docs (~1,500–3,000 docs)
- dbt official best-practices + dbt-core docs (~500–1,000 docs)
- Top 200 highly-upvoted dbt Discourse threads
- Total target: 3,000–5,000 documents, 50–150 MB raw text.

---

## 4. Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.11 | Matches Snowflake/dbt ecosystem |
| Crawler | httpx + trafilatura | Trafilatura beats BS4 for clean text |
| Chunking | langchain-text-splitters (fixed) + semantic-text-splitter | Two strategies for ablation |
| Embeddings | BAAI/bge-large-en-v1.5 | Open-source, beats ada-002 on MTEB |
| Vector DB | Qdrant in Docker | Free, fast, hybrid support |
| BM25 | Qdrant sparse vectors or rank_bm25 | Native hybrid |
| Reranker | BAAI/bge-reranker-large | Cross-encoder; most tutorials skip this |
| Query rewriter | GPT-4o-mini or Claude Haiku | Small, cheap |
| Generator | Claude Sonnet | Strong with citations |
| Eval | RAGAS | Industry-standard |
| Observability | Langfuse self-hosted | Free, trace-friendly |
| UI | Streamlit | Fast to build |
| Orchestration | make + docker-compose | One-line setup |
| Tests | pytest | Standard |
| CI | GitHub Actions | Eval gate lives here |

---

## 5. Repository structure

See the actual directory tree. Mirrors `src/erag/` layout from the brief.

---

## 6. Phased build plan (~50 hours)

- **Phase 0** — Setup (3 h): repo, pyproject.toml, docker-compose, stubs, README skeleton. ✅
- **Phase 1** — Corpus acquisition (6 h): crawlers, manifest.jsonl, corpus review.
- **Phase 2** — Indexing & retrieval (12 h): chunkers, embedder, Qdrant, hybrid RRF, rewriter, reranker, debug CLI.
- **Phase 3** — Generation + trust signals (6 h): prompts, answerer, Langfuse tracing.
- **Phase 4** — Evaluation harness (12 h): 50-question benchmark, RAGAS runner, ablation, failure analysis.
- **Phase 5** — Observability, CI, polish (8 h): Streamlit UI, GitHub Actions gate, README, blog post.

---

## 7. Definition of done

- [ ] Public GitHub repo with green CI badge
- [ ] `docker-compose up` works on a fresh clone
- [ ] `make crawl && make index && make eval` runs end-to-end
- [ ] 50-question benchmark in `evals/benchmark.yaml`, hand-curated, 4 tiers
- [ ] 3-config ablation table in `docs/ablation_results.md` with real numbers
- [ ] Failure analysis with 5+ entries in `eval/failure_analysis.md`
- [ ] `DECISIONS.md` with at least 8 entries
- [ ] README has headline metric, architecture diagram, "what didn't work" section
- [ ] GitHub Actions workflow fails when faithfulness < 0.80
- [ ] Streamlit UI shows cited sources, per-step latency, $ cost, faithfulness, debug toggle
- [ ] Langfuse screenshot in README
- [ ] Blog post draft in `docs/blog_post.md`
- [ ] No mention of "Production RAG" anywhere
