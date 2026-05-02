# Blog Post Draft

**Working title:** "Why my RAG was confidently wrong on 6 of 10 adversarial questions — and how I diagnosed it"

**Alternative title:** "What I learned hand-curating a 50-question RAG benchmark"

**Target length:** 700–1,200 words  
**Publish target:** Medium or dev.to; pin to LinkedIn Featured.

---

## Outline (to be written in Phase 5)

1. **Hook** — A specific adversarial question the system got confidently wrong. Quote the wrong answer. (3–4 sentences)

2. **What I built** — One paragraph: evaluated RAG over Snowflake + dbt docs, hybrid retrieval, hand-curated benchmark. Not "how to build a RAG" — that's not the angle.

3. **The benchmark lesson** — Why I hand-curated 50 questions instead of generating them with an LLM. The specific failure mode LLM-generated questions miss (they don't probe retrieval gaps, they probe semantic similarity).

4. **The adversarial tier finding** — What the adversarial tier revealed that the factual tier didn't. Specific example: a question that requires nuanced answer where the "obvious" answer is wrong.

5. **The hybrid retrieval finding** — BM25 wins on proper nouns. One concrete example: a query for a specific Snowflake feature flag where vector search returned semantically similar but wrong chunks, and BM25 found the exact doc.

6. **The failure I couldn't fix** — One honest failure from `failure_analysis.md` where the root cause is a corpus gap (information simply isn't in the docs). What that reveals about the limits of RAG.

7. **What I'd do differently** — Two specific things. Not generic advice.

8. **CTA** — Link to GitHub repo. Invite readers to try the benchmark questions.

---

*Fill in after Phase 4 eval run — the blog post should cite real numbers from ablation_results.md.*
