"""Central configuration — all knobs in one place, loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pydantic v2 settings

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"


class Settings(BaseSettings):
    # LLM providers
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="erag_docs", env="QDRANT_COLLECTION")

    # Langfuse
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", env="LANGFUSE_HOST")

    # Models
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    reranker_model: str = Field(default="BAAI/bge-reranker-large", env="RERANKER_MODEL")
    generator_model: str = Field(default="claude-sonnet-4-6", env="GENERATOR_MODEL")
    rewriter_model: str = Field(default="claude-haiku-4-5-20251001", env="REWRITER_MODEL")

    # Retrieval
    top_k_vector: int = Field(default=20, env="TOP_K_VECTOR")
    top_k_bm25: int = Field(default=20, env="TOP_K_BM25")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")
    rrf_k: int = Field(default=60, env="RRF_K")

    # Evaluation
    faithfulness_threshold: float = Field(default=0.80, env="FAITHFULNESS_THRESHOLD")

    # Crawl
    crawl_user_agent: str = Field(
        default="evaluated-rag-portfolio/0.1",
        env="CRAWL_USER_AGENT",
    )
    crawl_delay_secs: float = Field(default=1.0, env="CRAWL_DELAY_SECS")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
