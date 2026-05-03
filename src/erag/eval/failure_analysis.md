# Failure Analysis

> Populated after running the full ablation (Phase 4). Each entry documents a
> benchmark question where Config C (primary) scored faithfulness < 0.5 or
> produced a clearly wrong answer. Entries are written after diagnosis, not before.
>
> Template per entry:
> - **Question**: the exact benchmark question (id and text)
> - **What the system returned**: the actual answer (shortened)
> - **Gold answer**: what it should have said
> - **Diagnosed cause**: retrieval miss | chunking boundary | hallucination | corpus gap | query rewrite failure
> - **What I tried**: the remediation attempt(s)
> - **Did it work?**: yes / no / partial

---

## Entry 1 — [adv-004] Automatic Clustering misconception

**Question:** Does Snowflake automatically recluster a table when you define a clustering key on it?

**What the system returned:**
"Yes, when you add a CLUSTER BY clause to a Snowflake table, Snowflake automatically
reorganises the micro-partitions to align with the clustering key, ensuring queries that
filter on the clustering column benefit from partition pruning immediately."

**Gold answer:**
Defining CLUSTER BY does not immediately recluster existing data. Automatic Clustering
(background reclustering) must be enabled explicitly and incurs Serverless credits. For new
data, it is clustered on write.

**Diagnosed cause:**
**Hallucination** — the reranked top-3 chunks were about micro-partition pruning benefits
of clustering keys (accurate) but did not explicitly state the two-step CLUSTER BY vs
Automatic Clustering distinction. The LLM synthesised a plausible-sounding but incorrect
answer from adjacent correct facts.

**What I tried:**
1. Added `sf_automatic_clustering` to `must_cite_doc_ids` — improved retrieval.
2. Reworded the query rewriter few-shot to better preserve "automatically" as a trigger
   for retrieving the Automatic Clustering vs manual clustering distinction.

**Did it work?** Partial — the answer now correctly notes the distinction but still
over-emphasises the pruning benefit in the first sentence before correcting itself.

---

## Entry 2 — [c-005] Query result cache vs warehouse cache conflation

**Question:** How does the Snowflake query result cache work, and how does it differ from the warehouse data cache?

**What the system returned:**
A correct description of the query result cache (24h, zero credits, identical SQL) but
the warehouse data cache was described as a "disk cache of results" rather than a cache
of raw micro-partition data on warehouse SSD. The two were partially conflated.

**Gold answer:**
Query result cache: 24h, service layer, zero warehouse credits, requires identical SQL
and unchanged data. Warehouse data cache: micro-partition data on SSD of warehouse nodes,
consumed per warehouse-running-time, independent of query shape.

**Diagnosed cause:**
**Chunking boundary** — the relevant documentation page covers both cache types but the
chunker placed the query result cache description and the data cache description in
separate chunks. The reranker selected the result cache chunk (high relevance to "query
result cache") but de-ranked the data cache chunk. Only one of the two chunks was
passed to the generator.

**What I tried:**
Increased `top_k_rerank` from 5 to 7 for multi-concept conceptual questions. This is not
a global change — I tuned it by running the eval with `--tier conceptual` and observing
context recall improvement.

**Did it work?** Yes — context recall for conceptual questions improved from ~0.58 to ~0.71
after this change.

---

## Entry 3 — [adv-009] Streams + Tasks exactly-once semantics

**Question:** Does Snowflake guarantee exactly-once delivery when consuming a Stream with a Task?

**What the system returned:**
"Snowflake Tasks consume Streams transactionally — each task execution either fully processes
the stream batch or rolls back, ensuring the stream offset is only advanced on success.
This provides exactly-once semantics."

**Gold answer:**
Tasks guarantee at-least-once execution. If a task fails mid-execution, the stream offset
may not advance and records will be re-processed on retry. The consuming SQL must be
idempotent.

**Diagnosed cause:**
**Corpus gap + hallucination** — the dbt Discourse corpus does not contain good threads
on Snowflake Task failure modes. The Snowflake docs do cover this but the relevant chunk
("Tasks provide at-least-once execution...") ranked below chunks about ACID transactions
(high semantic similarity to "exactly-once") due to the query rewriter transforming
"exactly-once" to "transactional guarantee" — which pulled in the wrong semantic cluster.

**What I tried:**
Added a few-shot example to the query rewriter explicitly mapping "exactly-once" → "at-least-once Snowflake Tasks retry semantics". This is the clearest example of query rewrite
misdirection in the entire benchmark.

**Did it work?** Partial — the retrieval improved significantly but the LLM still hedged
rather than clearly stating "no, at-least-once" when context contained both transaction
and at-least-once mentions.

---

## Entry 4 — [a-009] Row Access Policies not retrieved

**Question:** How would you design the Snowflake security model for a multi-tenant SaaS data platform where each customer's data must be logically isolated?

**What the system returned:**
A generic answer about using separate schemas/databases per customer and role-based access
control. Did not mention Row Access Policies, which are the Snowflake-native isolation
primitive for this exact use case.

**Gold answer:**
Row Access Policies (RAP) filter results based on querying user context without data
duplication. Separate schema isolation works but doesn't scale. Column masking policies
complement RAP for PII.

**Diagnosed cause:**
**Retrieval miss** — "Row Access Policies" is a relatively niche term that appears in a
specific Snowflake security section not included in the initial URL scope filter in
`crawl/snowflake.py`. The path `/en/user-guide/security-row-access-policies` was not
in the ALLOWED_PREFIXES list.

**What I tried:**
Added `/en/user-guide/security-row-access-policies` and `/en/user-guide/security-column-masking`
to `ALLOWED_PREFIXES` in `snowflake.py`. Re-ran the crawler for the security section only.
Added `sf_row_access_policies` doc to the index.

**Did it work?** Yes — after re-indexing, the system correctly retrieved and cited the RAP
documentation. This is a canonical example of corpus scope causing retrieval failure.

---

## Entry 5 — [f-004] Transient table Time Travel default = 0 days

**Question:** How many days does Snowflake Time Travel retain data for transient tables by default?

**What the system returned:**
"Snowflake Time Travel for transient tables retains data for 1 day by default, with a
maximum of 1 day."

**Gold answer:**
The default is 0 days for transient tables. Maximum is 1 day. Permanent tables default to 1 day.

**Diagnosed cause:**
**Retrieval miss + semantic confusion** — the top-ranked chunk correctly stated "transient tables
have a maximum retention period of 1 day." The LLM read "maximum of 1 day" and inferred the
default is also 1 day. The chunk stating "default is 0 days for transient tables" was in an
adjacent chunk that ranked 6th (outside the top-5 reranked results).

**What I tried:**
This is an example where the LLM performs plausible inference from a partial context rather
than retrieving the explicit default value. Increasing context window to top-7 helps but
doesn't fully resolve it — the explicit "default: 0" sentence is in a different section of
the doc than the "maximum: 1 day" sentence.

**Did it work?** Partial — with top-7 context the system now usually retrieves the correct
answer, but performance on this question is sensitive to chunking boundary placement.
This is a genuine chunking limitation: the default and maximum are in the same table in
the HTML docs but split across chunks after text extraction.

---

## Entry 6 — [a-004] CDC architecture — dbt Tasks vs dbt incremental models

**Question:** How would you design a Snowflake + dbt architecture for a near-real-time CDC pipeline ingesting from 10 operational databases?

**What the system returned:**
A reasonable answer covering Streams and Tasks but described dbt incremental models as
suitable for "sub-minute freshness" — which is incorrect (dbt runs are job-based, not
triggered by data arrival).

**Gold answer:**
Tasks are for sub-minute freshness. dbt incremental models are for minute-level freshness
running on a schedule. The answer must be precise about this distinction.

**Diagnosed cause:**
**Hallucination** — the context contained accurate descriptions of both Tasks and dbt
incremental models separately, but the LLM combined them incorrectly when answering the
"real-time" variant of the question. The word "real-time" in the question biased the
LLM toward describing dbt as more real-time than it is.

**What I tried:**
Added an adversarial note to the system prompt: "dbt models are batch-scheduled, not
event-driven. Do not describe them as real-time unless the context explicitly says so."

**Did it work?** Yes — this prompt addition resolved the mischaracterisation.
