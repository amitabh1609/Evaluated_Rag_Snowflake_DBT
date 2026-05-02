"""CLI entry point: chunk → embed → upsert all documents into Qdrant.

Usage:
    python -m erag.index.build_index [--chunker fixed|semantic] [--reset]
"""

# TODO Phase 2: implement
#   - Load manifest.jsonl from data/raw/
#   - For each document: chunk → embed dense + sparse → upsert to Qdrant
#   - Progress bar via tqdm
#   - Idempotent: skip already-indexed doc_ids unless --reset passed
