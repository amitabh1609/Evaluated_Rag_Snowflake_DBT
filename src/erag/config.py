"""Central configuration — all knobs in one place, loaded from environment."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"


class Settings(BaseSettings):
    # LLM providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "erag_docs"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Models
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    generator_model: str = "claude-sonnet-4-6"
    rewriter_model: str = "claude-haiku-4-5-20251001"

    # Retrieval
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60

    # Evaluation
    faithfulness_threshold: float = 0.80

    # Crawl
    crawl_user_agent: str = "evaluated-rag-portfolio/0.1"
    crawl_delay_secs: float = 1.0

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
