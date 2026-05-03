"""LLM-assisted query rewriting.

Rewrites vague developer questions into precise search queries that
perform better against technical documentation corpora.

Design choices:
- Uses Claude Haiku (or gpt-4o-mini) — cheap, fast, good enough for rewriting.
- Results are cached with lru_cache keyed on the raw question string, so
  repeated questions (common in eval runs) don't incur API calls.
- Original + rewritten query are logged to Langfuse as a span.
- Falls back to the original query on any API error.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You rewrite vague developer questions into precise search queries for technical documentation.

Rules:
1. Preserve all proper nouns exactly: Snowflake feature names, dbt commands, version numbers,
   configuration parameter names (e.g. AUTO_SUSPEND, dbt run --select, SHOW WAREHOUSES).
2. Expand abbreviations when the full form is more searchable.
3. Remove conversational filler ("can you tell me", "I want to know", "how do I").
4. If the question contains multiple sub-questions, focus on the most specific one.
5. Output ONE rewritten query and nothing else. No explanation, no punctuation at the end.

Examples:
User: what's the deal with auto suspend on snowflake warehouses
Assistant: Snowflake warehouse AUTO_SUSPEND default value configuration

User: when should i pick dynamic tables over streams in my dbt project
Assistant: Snowflake Dynamic Tables vs Streams comparison use cases CDC

User: my dbt tests are really slow how do i speed them up
Assistant: dbt test performance optimization reduce test runtime strategies
"""


@lru_cache(maxsize=512)
def rewrite_query(question: str, model: str = "") -> str:
    """Return a rewritten search query for `question`.

    Falls back to the original question on any API error.
    """
    from erag.config import settings

    _model = model or settings.rewriter_model

    try:
        if _model.startswith("claude"):
            return _rewrite_anthropic(question, _model)
        else:
            return _rewrite_openai(question, _model)
    except Exception as e:
        logger.warning("query rewrite failed (%s) — using original: %s", e, question)
        return question


def _rewrite_anthropic(question: str, model: str) -> str:
    import anthropic
    from erag.config import settings

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    rewritten = msg.content[0].text.strip()
    logger.debug("rewrite: %r → %r", question, rewritten)
    return rewritten


def _rewrite_openai(question: str, model: str) -> str:
    from openai import OpenAI
    from erag.config import settings

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=128,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.debug("rewrite: %r → %r", question, rewritten)
    return rewritten
