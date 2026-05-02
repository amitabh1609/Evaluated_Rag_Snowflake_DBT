"""Semantic chunking: split at embedding-similarity boundaries.

Used for Config C (primary) in the ablation.
Relies on semantic-text-splitter backed by sentence-transformers.
"""

# TODO Phase 2: implement
#   from semantic_text_splitter import TextSplitter
#   from sentence_transformers import SentenceTransformer
#
#   def chunk_document(doc_id: str, text: str, max_tokens: int = 512) -> list[Chunk]:
#       ...
