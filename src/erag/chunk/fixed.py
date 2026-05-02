"""Fixed-size chunking: 512 tokens, 64-token overlap.

Used for Config A (baseline) and Config B in the ablation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# TODO Phase 2: implement
#   from langchain_text_splitters import RecursiveCharacterTextSplitter
#   splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#       model_name="cl100k_base",
#       chunk_size=512,
#       chunk_overlap=64,
#   )
#
#   def chunk_document(doc_id: str, text: str) -> list[Chunk]:
#       ...
