"""Langfuse observability client.

Wraps the Langfuse SDK to provide consistent span naming and graceful
no-op behaviour when keys are not configured (e.g. in CI).

Every query is traced as a top-level generation with child spans:
  rewrite → embed → retrieve → rerank → generate

Token counts and cost estimates are attached to the generate span.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── No-op stubs ───────────────────────────────────────────────────────────────
# Used when Langfuse is not configured so the rest of the code never branches.


class _NoOpSpan:
    def end(self, output: Any = None, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class _NoOpTrace:
    def span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def generation(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass


# ── Real client ───────────────────────────────────────────────────────────────

_client = None  # langfuse.Langfuse singleton


def _get_client():
    global _client
    if _client is not None:
        return _client

    from erag.config import settings

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.debug("Langfuse keys not set — tracing disabled")
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled → %s", settings.langfuse_host)
    except Exception as e:
        logger.warning("Langfuse init failed (%s) — tracing disabled", e)
        return None

    return _client


def start_trace(name: str = "rag_query", **kwargs: Any):
    """Start a new Langfuse trace. Returns a no-op trace if Langfuse is unavailable."""
    client = _get_client()
    if client is None:
        return _NoOpTrace()
    try:
        return client.trace(name=name, **kwargs)
    except Exception as e:
        logger.debug("Langfuse trace creation failed: %s", e)
        return _NoOpTrace()


def flush() -> None:
    """Flush any buffered Langfuse events (call at end of eval runs)."""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


@contextmanager
def span(trace, name: str, input: Any = None) -> Generator[Any, None, None]:
    """Context manager for a Langfuse span. Ends the span on exit."""
    s = trace.span(name=name, input=input)
    try:
        yield s
    finally:
        s.end()
