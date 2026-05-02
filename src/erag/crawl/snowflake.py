"""Crawl Snowflake architecture, warehouse, and table-format documentation.

Respects robots.txt, delays 1 req/sec, identifies via CRAWL_USER_AGENT.
Outputs clean markdown files to data/raw/snowflake/ and appends to manifest.jsonl.
"""

# TODO Phase 1: implement polite crawler
#   - Parse sitemap or seed URL list
#   - httpx async client with rate limiting (asyncio.sleep between requests)
#   - trafilatura.extract() for clean text
#   - Write data/raw/snowflake/<doc_id>.md
#   - Append {"doc_id": ..., "url": ..., "title": ..., "source": "snowflake"} to manifest.jsonl
