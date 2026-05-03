"""Dense embeddings via BAAI/bge-large-en-v1.5 and BM25 index construction.

BGE note: BGE models expect a query instruction prefix for *queries* only,
not for documents at index time. Docs are embedded as plain text.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi
    from erag.chunk import Chunk

logger = logging.getLogger(__name__)

# BGE-large instruction prefix — prepended to queries only
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 1024  # BGE-large-en-v1.5 output dimension

_model: SentenceTransformer | None = None


def get_model(model_name: str = "BAAI/bge-large-en-v1.5") -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("loading embedding model %s …", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def embed_documents(texts: list[str], model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 64) -> np.ndarray:
    """Embed document texts — NO query instruction prefix."""
    model = get_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query: str, model_name: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """Embed a single query — WITH BGE instruction prefix."""
    model = get_model(model_name)
    return model.encode(
        BGE_QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ── BM25 ──────────────────────────────────────────────────────────────────────

_BM25_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize_bm25(text: str) -> list[str]:
    """Simple whitespace/punctuation tokeniser for BM25. Lowercased."""
    return _BM25_PATTERN.findall(text.lower())


def build_bm25_index(chunks: list["Chunk"]) -> "BM25Okapi":
    from rank_bm25 import BM25Okapi
    logger.info("building BM25 index over %d chunks …", len(chunks))
    corpus = [tokenize_bm25(c.text) for c in chunks]
    return BM25Okapi(corpus)


def save_bm25(index: "BM25Okapi", chunk_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": index, "chunk_ids": chunk_ids}, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 index saved → %s", path)


def load_bm25(path: Path) -> tuple["BM25Okapi", list[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]
