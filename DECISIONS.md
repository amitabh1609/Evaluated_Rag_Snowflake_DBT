# Architecture Decisions

Each entry follows this template:

```
## Decision: <name>
**Date:** YYYY-MM-DD
**Options considered:** A, B, C
**Choice:** B
**Why (3 sentences):** ...
**What I'd reconsider (1 sentence):** ...
```

---

## Decision: Vector DB — Qdrant over Pinecone / Weaviate / pgvector

**Date:** 2026-05-02
**Options considered:** Pinecone (managed), Weaviate (Docker), pgvector (Postgres extension), Qdrant (Docker)
**Choice:** Qdrant
**Why (3 sentences):** Qdrant runs entirely in Docker with a single image, eliminating cloud
vendor lock-in and keeping the project free to reproduce on any laptop. It natively supports
both dense (float32) and sparse (uint32) vectors in the same collection, which is exactly what
hybrid vector + BM25 retrieval needs without a separate BM25 index service. pgvector was
tempting given existing Postgres familiarity, but it lacks first-class sparse-vector support and
its HNSW performance at 50k+ vectors trails Qdrant in benchmarks.
**What I'd reconsider (1 sentence):** If this were a production multi-tenant service with
>10M vectors and a dedicated ops team, Pinecone's managed scaling would be worth the cost.

---

## Decision: Embedding model — BGE-large-en-v1.5 over OpenAI text-embedding-ada-002

**Date:** 2026-05-02
**Options considered:** OpenAI text-embedding-ada-002, OpenAI text-embedding-3-small,
BAAI/bge-large-en-v1.5, intfloat/e5-large-v2
**Choice:** BAAI/bge-large-en-v1.5
**Why (3 sentences):** BGE-large ranks in the top tier of the MTEB leaderboard and
outperforms ada-002 on retrieval tasks, particularly for technical English prose. Running
it locally via sentence-transformers eliminates per-embedding API cost, which matters for
re-indexing experiments during the ablation phase. Choosing an open-source model also signals
to interviewers that I evaluated alternatives rather than defaulting to the cheapest API call.
**What I'd reconsider (1 sentence):** text-embedding-3-large now beats BGE-large on several
MTEB subsets; if latency of local inference becomes a bottleneck, switching to the OpenAI v3
family is a clean drop-in.

---

## Decision: Hybrid retrieval — vector + BM25 + RRF over vector-only

**Date:** 2026-05-02
**Options considered:** Vector-only (dense ANN), BM25-only, vector + BM25 concatenated,
vector + BM25 merged via Reciprocal Rank Fusion (RRF)
**Choice:** Hybrid with RRF
**Why (3 sentences):** Technical documentation contains many exact-match terms (Snowflake
feature names, dbt command flags, version numbers) where BM25 consistently outperforms dense
retrieval because semantic similarity is noisy for rare proper nouns. RRF is a parameter-light
fusion method — k=60 is the standard constant derived from Cormack et al. 2009 — that avoids
the score-normalization headache of combining cosine similarities with BM25 scores. The
ablation (Config A vs C) is specifically designed to quantify the precision gain, making the
choice empirically defensible rather than just principled.
**What I'd reconsider (1 sentence):** Learned sparse models (SPLADE, BM42) may eventually
replace explicit BM25; if Qdrant's native sparse-vector support matures further I'd switch.

---

## Decision: Chunking strategy — semantic over fixed (pending ablation confirmation)

**Date:** 2026-05-02
**Options considered:** Fixed 512-token with 64-token overlap, sentence-level, semantic
(embedding-similarity boundary detection via semantic-text-splitter)
**Choice:** Semantic chunking for Config C (primary); fixed for Config A/B (ablation baseline)
**Why (3 sentences):** Documentation sections are not uniformly dense — a "Limitations"
paragraph sits next to a 50-row reference table — so fixed-size windows routinely split
coherent concepts across chunk boundaries. Semantic chunking uses embedding-similarity drops
to find natural boundaries, keeping related sentences together and reducing the chance that
a gold answer spans two chunks. The ablation is the only honest test; if Config B (fixed +
reranker) beats Config C on context precision, I will update this decision.
**What I'd reconsider (1 sentence):** Semantic chunking is slower at index time and harder
to reproduce exactly; if the ablation delta is <2 percentage points I'll revert to fixed for
simplicity.

---

## Decision: Hand-curated benchmark over LLM-generated questions

**Date:** 2026-05-02
**Options considered:** GPT-4o to generate 50 question-answer pairs, LLM + human review,
fully hand-curated
**Choice:** Fully hand-curated
**Why (3 sentences):** LLM-generated questions inherit the retrieval model's blind spots —
an LLM asked to generate "hard" questions over the same corpus it was trained on will produce
questions the system is likely to answer correctly, inflating eval scores. Hand-curation forces
me to actually read the corpus, which is the only way to write genuine adversarial and
architectural questions that probe failure modes. The benchmark is the spine of the project's
credibility claim; cutting corners here undermines every metric in the README.
**What I'd reconsider (1 sentence):** For scaling beyond 50 questions, a human-in-the-loop
LLM pipeline (generate → human approve → human rewrite gold answer) would be a defensible
compromise.

---

## Decision: Langfuse over LangSmith / Phoenix / Arize

**Date:** 2026-05-02
**Options considered:** LangSmith (managed), Phoenix (local), Arize (managed), Langfuse (self-hosted)
**Choice:** Langfuse self-hosted via docker-compose
**Why (3 sentences):** Langfuse's docker-compose setup is one YAML file, making it trivially
reproducible on any machine — a reviewer cloning this repo gets the full observability stack
for free. LangSmith requires a paid account above the free tier and is not self-hostable;
Phoenix has a smaller community and less polished trace UI. Langfuse's SDK records spans,
token counts, and cost estimates out of the box, which feeds directly into the latency + cost
panel in the Streamlit UI.
**What I'd reconsider (1 sentence):** If the team already uses LangSmith in production, the
integration parity would make LangSmith the obvious choice despite the cost.

---

## Decision: "Evaluated RAG" not "Production RAG"

**Date:** 2026-05-02
**Options considered:** "Production RAG System", "Evaluated RAG System", "RAG Pipeline"
**Choice:** "Evaluated RAG System"
**Why (3 sentences):** "Production RAG" is a claim that invites the interview question
"Is it serving real traffic?" — to which the honest answer is no, and the credibility gap
undermines everything else. "Evaluated" is precise and defensible: the system has a benchmark,
reproducible metrics, and a CI gate, all of which can be demonstrated on demand. The word
"evaluated" signals engineering maturity — knowing how to measure a system is harder and rarer
than building one.
**What I'd reconsider (1 sentence):** Nothing; this naming choice is correct and I would
not change it.

---

## Decision: dbt Discourse threads in the corpus

**Date:** 2026-05-02
**Options considered:** Snowflake + dbt official docs only, add Stack Overflow, add dbt Discourse
**Choice:** Add top-200 dbt Discourse threads (sorted by likes, last 3 years)
**Why (3 sentences):** Official documentation explains what features do; practitioners on
Discourse explain what goes wrong, which configuration works in practice, and how to work
around edge cases — exactly the knowledge a senior engineer needs. The top-200 liked threads
are a high-signal, low-noise subset: high likes correlate with questions many people had and
answers that actually worked. Including Discourse is the single clearest differentiator from
tutorial RAG projects that index only the vendor docs.
**What I'd reconsider (1 sentence):** Stack Overflow would add signal too, but scraping SO
at scale risks ToS issues; Discourse's public API is clean and explicitly allows crawling.
