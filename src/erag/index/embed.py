"""Compute dense embeddings using BAAI/bge-large-en-v1.5.

Batches chunks, returns float32 numpy arrays.
"""

# TODO Phase 2: implement
#   from sentence_transformers import SentenceTransformer
#
#   BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
#
#   def embed_batch(texts: list[str], model: SentenceTransformer) -> np.ndarray:
#       # BGE models want an instruction prefix for queries (not for documents)
#       ...
