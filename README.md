# Evaluated RAG System for Snowflake & dbt Documentation

![Eval Gate](https://github.com/amitabh1609/Evaluated_Rag_Snowflake_DBT/actions/workflows/eval.yml/badge.svg)

> **Headline metric (Config C — primary):**
> Faithfulness **≥ 0.80** on a 50-question, 4-tier hand-curated benchmark.
> Hybrid retrieval (BGE-large + BM25 + RRF k=60) + BGE cross-encoder reranking.
> P95 latency ~4–6 s, estimated cost ~$0.002–$0.005 per query.

---

## What this is

A question-answering system over Snowflake and dbt documentation, built to be
_measurably correct_ rather than just functional. The system uses hybrid retrieval
(dense vector search + BM25 merged via Reciprocal Rank Fusion), a BGE-large
cross-encoder reranker, and Claude Sonnet as the generator. Every answer is
grounded in cited source chunks. Quality is tracked with four RAGAS metrics across
a hand-curated 50-question benchmark that was written specifically to catch the
failure modes that matter — precision-sensitive facts, multi-document synthesis,
open architecture questions, and adversarial traps.

The repo is a portfolio artifact. The goal was not to build a product but to
demonstrate that I can build and rigorously evaluate a retrieval system, diagnose
its failures, and document the process honestly.

---

## Architecture

```
Corpus (Snowflake docs + dbt docs + dbt Discourse)
    └─► crawl (trafilatura, robots.txt, rate-limited)
        └─► chunk (fixed-512 OR semantic via bert-base-uncased)
            └─► embed (BAAI/bge-large-en-v1.5, 1024-dim)
                └─► Qdrant (ANN) + BM25 pickle (rank_bm25)
                        │
         query ─► rewrite (Claude Haiku, 3-shot, lru_cache)
                  └─► embed query (BGE query-instruction prefix)
                      ├─► vector_search (top-20)
                      ├─► bm25_search  (top-20)
                      └─► RRF merge k=60
                          └─► rerank (BAAI/bge-reranker-large)
                              └─► generate (Claude Sonnet, citation format)
                                  └─► RAGAS eval + Langfuse trace
```

---

## Why this corpus

Snowflake and dbt documentation are good stress tests for a RAG system because:

- **Precision traps**: Snowflake has many pairs of features that sound similar but
  differ sharply (Time Travel vs Fail-safe, CLUSTER BY vs Automatic Clustering,
  Streams vs Dynamic Tables). A system that retrieves adjacent-but-wrong chunks
  will hallucinate plausible-sounding incorrect answers.
- **Multi-document synthesis**: answering "how do Streams and dbt incremental models
  compare for CDC?" requires retrieving from Snowflake docs and dbt docs simultaneously.
- **Community signal**: dbt Discourse top-200 threads capture practitioner knowledge
  that is not in official docs — e.g., the actual production patterns people use for
  incremental model strategies.

---

## Search merge logic

Each query runs two independent retrievers:

1. **Dense vector search** — query embedded with `BAAI/bge-large-en-v1.5` using the
   BGE query-instruction prefix; ANN search in Qdrant (cosine), top-20.
2. **BM25** — rank_bm25 `BM25Okapi`, tokenised on whitespace, serialised as a pickle;
   top-20 by BM25 score.

The two ranked lists are merged via **Reciprocal Rank Fusion**:

```
score(d) = Σᵢ  1 / (k + rankᵢ(d))    k = 60
```

k=60 is the empirically validated default from Cormack, Clarke & Buettcher (2009).
It compresses rank differences so a result that ranks 1st in one list and absent in
the other scores similarly to a result that ranks 3rd and 5th — which is exactly
the right behaviour when neither retriever has unambiguous authority.

After RRF merge the top-20 candidates are reranked by `BAAI/bge-reranker-large`
(a cross-encoder), which scores the full query–chunk pair. Top-5 reranked chunks
are passed to the generator.

---

## Evaluation methodology

### Benchmark design

50 questions across 4 tiers, written by hand to probe real failure modes:

| Tier | N | Design intent |
|------|---|---------------|
| Factual | 15 | Precision-sensitive facts with common misconception traps |
| Conceptual | 15 | Require synthesising 2–3 documents |
| Architectural | 10 | Open-ended design questions; no single correct answer |
| Adversarial | 10 | Questions designed to elicit confident wrong answers |

Key adversarial examples:
- "Does Snowflake automatically recluster a table when you define a clustering key?"
  (trap: CLUSTER BY ≠ Automatic Clustering)
- "Does Snowflake guarantee exactly-once delivery when consuming a Stream with a Task?"
  (trap: at-least-once, not exactly-once)
- "Is dbt an orchestration tool?" (trap: dbt is a transformation tool, not an orchestrator)

### RAGAS metrics

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Does the answer stay within the retrieved context? |
| Answer Relevancy | Is the answer relevant to the question? |
| Context Precision | Are the retrieved chunks actually useful for the question? |
| Context Recall | Did we retrieve the chunks needed to answer correctly? |

CI gate: `faithfulness < 0.80` → build fails.

---

## Ablation results

Three configurations tested on the full 50-question benchmark:

| Config | Chunking | Retrieval | Reranker | Faithfulness | Ctx Recall |
|--------|----------|-----------|----------|-------------|------------|
| A | fixed-512 | vector-only | none | baseline | baseline |
| B | fixed-512 | vector-only | BGE-rerank-L | +Δ | +Δ |
| **C** | **semantic** | **hybrid+RRF** | **BGE-rerank-L** | **primary** | **primary** |

Full numbers including per-tier breakdown: [docs/ablation_results.md](docs/ablation_results.md).

Key finding: hybrid retrieval (B→C) improved context recall most on conceptual and
adversarial questions — exactly the tiers where BM25's keyword precision helps.
Semantic chunking over fixed-512 reduced chunking-boundary failures on multi-concept
questions (see failure analysis Entry 2).

---

## Failure analysis

Six diagnosed failure cases with root cause and remediation:
[src/erag/eval/failure_analysis.md](src/erag/eval/failure_analysis.md)

Summary:

| ID | Root cause | Fixed? |
|----|-----------|--------|
| adv-004 | Hallucination from adjacent (correct) clustering facts | Partial |
| c-005 | Chunking boundary split two cache types | Yes |
| adv-009 | Query rewrite misdirected "exactly-once" | Partial |
| a-009 | Corpus scope — Row Access Policies not crawled | Yes |
| f-004 | LLM inferred default=max from partial context | Partial |
| a-004 | Hallucination — dbt described as real-time | Yes |

---

## Latency and cost

Observed on Config C (semantic chunks, hybrid+RRF, BGE reranker, Claude Sonnet):

| Step | Typical latency |
|------|----------------|
| Query rewrite (Haiku) | ~300–600 ms |
| Embed query (BGE-large) | ~80–150 ms |
| Vector + BM25 + RRF | ~50–100 ms |
| BGE reranker (top-20→top-5) | ~200–500 ms |
| Generate (Claude Sonnet) | ~2–4 s |
| **Total P50** | **~3–5 s** |

Estimated cost per query: ~$0.002–$0.005 (dominated by Claude Sonnet generation).

---

## What didn't work

### 1. `from_tiktoken_encoder(model_name=)` instead of `encoding_name=`

Passed `"cl100k_base"` as `model_name` to `RecursiveCharacterTextSplitter.from_tiktoken_encoder`.
Got `KeyError: 'Could not automatically map cl100k_base to a tokeniser'` because tiktoken's
model→encoding map expects model names like `gpt-4`, not encoding names. Fixed by switching
to `encoding_name="cl100k_base"`.

Lesson: read the parameter chain (`encoding_for_model` vs `get_encoding`) before assuming
parameter names map 1:1 across library boundaries.

### 2. Citation parser double-encoding chunk_id

The citation regex captured `doc_id=sf_wh` and `chunk_suffix=3` separately, then looked
up `chunk_suffix` directly in the payload dict — always missing, because the dict is keyed
by `"sf_wh:3"`. A second bug: the lookup builder was inserting `"sf_wh:sf_wh:3"` via
a redundant f-string.

Fix: reconstruct `full_chunk_id = f"{doc_id}:{chunk_suffix}"` before the lookup.
Lesson: write the citation unit test _before_ the integration test.

### 3. Query rewrite misdirection on "exactly-once"

The query rewriter's few-shot rewrote "exactly-once" → "transactional guarantee", which
pulled in ACID transaction chunks (high semantic similarity) and suppressed the correct
at-least-once task-retry chunk. Adding a negative few-shot example ("exactly-once" →
"at-least-once Snowflake Tasks retry semantics") improved retrieval but the generator
still hedged rather than clearly stating the correct answer.

Lesson: a rewriter that is too aggressive about semantic normalisation can hurt precision
more than it helps recall.

### 4. LLM inference from partial context (f-004 transient table default)

The top-ranked chunk correctly stated "transient tables have a maximum retention period
of 1 day." The LLM read this and inferred the default is also 1 day. The chunk containing
"default is 0 days" ranked 6th (outside top-5). Increasing top_k_rerank to 7 helped
but chunking boundary placement made this sensitive.

Lesson: for precision-sensitive fact questions, the explicit statement of defaults and
maximums should be in the same chunk — or the context window should be large enough to
include both.

### 5. Corpus scope gap (a-009 Row Access Policies)

Row Access Policies weren't in the initial crawl scope (`ALLOWED_PREFIXES` in
`snowflake.py`). The system gave a generic RBAC answer that missed the Snowflake-native
isolation primitive entirely. Fixed by adding the security section URLs and re-crawling.

Lesson: benchmark questions should be written before finalising the crawl scope, so
corpus gaps surface early.

---

## What I'd do differently

1. **Write adversarial benchmark questions first, then build the crawl scope around them.**
   Starting with corpus collection then benchmarking meant I discovered scope gaps late.

2. **Add a per-chunk provenance test to the benchmark.**
   A test that verifies each benchmark question has at least one answer-bearing chunk
   in the index catches corpus gaps without running the full eval.

3. **Use a smarter chunking strategy per doc type.**
   Snowflake docs have HTML tables with exact numeric values (defaults, limits). The
   current chunker splits these at sentence boundaries, separating the "default" column
   from its value. A table-aware extractor would eliminate the f-004 class of failures.

4. **Evaluate the query rewriter independently.**
   The rewriter is currently evaluated only via downstream RAGAS metrics. A dedicated
   rewrite quality test (does the rewritten query retrieve the right doc?) would catch
   misdirection earlier.

---

## Run it locally

Requirements: Python 3.11+, Docker, `uv`

```bash
git clone https://github.com/amitabh1609/Evaluated_Rag_Snowflake_DBT
cd Evaluated_Rag_Snowflake_DBT
cp .env.example .env          # fill in ANTHROPIC_API_KEY + OPENAI_API_KEY
make install                  # uv sync — installs all pinned deps
make up                       # starts Qdrant + Langfuse via docker-compose
make crawl                    # ~1–2 hours (polite, rate-limited crawl)
make index                    # embed + upsert into Qdrant + build BM25 index
make ui                       # Streamlit UI on http://localhost:8501
```

Run the evaluation:

```bash
make eval-smoke               # 10-question CI subset (~5 min)
make eval                     # full 50-question benchmark (~60 min)
make ablation                 # 3-config ablation (~90 min)
```

Debug a single query:

```bash
make debug ARGS='"How does Time Travel differ for transient tables?"'
```

---

## Project structure

```
src/erag/
  config.py          — pydantic-settings: all knobs in one place
  crawl/             — trafilatura-based crawlers (Snowflake, dbt docs, dbt Discourse)
  chunk/             — fixed-512 and semantic chunkers
  index/             — BGE-large embedder, Qdrant client, BM25 builder
  retrieve/          — vector, BM25, hybrid RRF, query rewriter, BGE reranker, debug
  generate/          — system prompts, citation parser, cost estimator, RagAnswer
  eval/              — RAGAS runner, 3-config ablation, 50-question benchmark, failure analysis
  observe/           — Langfuse client with _NoOp fallback
  ui/                — Streamlit app
tests/               — chunking, hybrid RRF, guardrails, eval smoke
docs/                — ablation_results.md, blog_post.md
.github/workflows/   — eval.yml: CI gate on faithfulness ≥ 0.80
```

---

## Decisions log

Every non-obvious architectural choice — vector DB selection, embedding model,
BM25 library, chunking strategy, benchmark design — is documented with its
tradeoffs in [DECISIONS.md](DECISIONS.md).
