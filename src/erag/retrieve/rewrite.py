"""LLM-assisted query rewriting.

Rewrites vague developer questions into precise search queries for technical docs.
Uses claude-haiku-4-5 (or gpt-4o-mini) — cheap, fast, good enough for rewriting.

System prompt:
    "You rewrite vague developer questions into precise search queries for technical
    documentation. Preserve all proper nouns (Snowflake feature names, dbt commands,
    version numbers). Output one rewritten query and nothing else."

Rewrites are cached in-process (questions repeat in eval runs).
Original + rewritten query are logged to Langfuse.
"""

# TODO Phase 2: implement
#   - Build SYSTEM_PROMPT with 3 few-shot examples
#   - @functools.lru_cache on the rewrite function keyed by original query
#   - Log to Langfuse span
