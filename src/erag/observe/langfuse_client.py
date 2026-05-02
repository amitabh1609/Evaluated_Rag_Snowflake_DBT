"""Langfuse tracing client — thin wrapper for consistent span naming.

Every query is traced as a top-level generation with child spans:
  rewrite → embed → retrieve → rerank → generate

Token counts and cost estimates are attached to the generate span.
"""

# TODO Phase 3: implement
#   from langfuse import Langfuse
#
#   _client: Langfuse | None = None
#
#   def get_client() -> Langfuse:
#       global _client
#       if _client is None:
#           _client = Langfuse(...)
#       return _client
#
#   def trace_query(question: str) -> langfuse.client.StatefulTraceClient: ...
