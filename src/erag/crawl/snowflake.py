"""Crawl focused sections of the Snowflake documentation.

Scope (brief requirement: ~1,500–3,000 docs):
  - Virtual warehouses & compute
  - Tables (standard, clustering, Iceberg/table formats)
  - Dynamic Tables
  - Streams & Tasks (CDC building blocks)
  - Transactions & isolation
  - Architecture & concepts
  - Key SQL reference pages (DDL relevant to the above)

Strategy: parse docs.snowflake.com sitemap index → filter to allowed path prefixes
→ crawl each URL → trafilatura extraction → save as markdown.

Usage:
    python -m erag.crawl.snowflake [--limit N] [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from erag.config import RAW_DIR, settings
from erag.crawl.base import BaseCrawler, CrawledDoc, url_to_doc_id

logger = logging.getLogger(__name__)

# ── Scope filter ─────────────────────────────────────────────────────────────
# Only crawl URLs whose path starts with one of these prefixes.
ALLOWED_PREFIXES = (
    "/en/user-guide/warehouses",
    "/en/user-guide/tables",
    "/en/user-guide/dynamic-tables",
    "/en/user-guide/streams",
    "/en/user-guide/tasks",
    "/en/user-guide/transactions",
    "/en/user-guide/data-sharing",
    "/en/user-guide/clustering",
    "/en/user-guide/micropartitions",
    "/en/user-guide/querying",
    "/en/user-guide/query-profile",
    "/en/user-guide/query-history",
    "/en/user-guide/cost",
    "/en/user-guide/credit",
    "/en/user-guide/security-access-control",
    "/en/concepts",
    "/en/guides-overview",
    # SQL reference — only DDL/DML for the in-scope objects
    "/en/sql-reference/sql/create-warehouse",
    "/en/sql-reference/sql/alter-warehouse",
    "/en/sql-reference/sql/create-table",
    "/en/sql-reference/sql/alter-table",
    "/en/sql-reference/sql/create-dynamic-table",
    "/en/sql-reference/sql/create-stream",
    "/en/sql-reference/sql/create-task",
    "/en/sql-reference/sql/begin",
    "/en/sql-reference/sql/commit",
    "/en/sql-reference/sql/rollback",
    "/en/sql-reference/sql/show-warehouses",
    "/en/sql-reference/parameters",
    "/en/sql-reference/functions/system",
    # Iceberg / open table formats
    "/en/user-guide/tables-iceberg",
    "/en/user-guide/data-load",
)

SITEMAP_INDEX = "https://docs.snowflake.com/sitemap_index.xml"
BASE_URL = "https://docs.snowflake.com"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _path_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    path = urlparse(url).path
    return any(path.startswith(p) for p in ALLOWED_PREFIXES)


async def _parse_sitemap_index(crawler: BaseCrawler, client: httpx.AsyncClient) -> list[str]:
    """Return list of child sitemap URLs from the sitemap index."""
    xml = await crawler.fetch_xml(client, SITEMAP_INDEX)
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
        return [loc.text for loc in root.findall(".//sm:loc", NS) if loc.text]
    except ET.ParseError as e:
        logger.error("sitemap index parse error: %s", e)
        return []


async def _parse_sitemap(crawler: BaseCrawler, client: httpx.AsyncClient, url: str) -> list[str]:
    """Return doc URLs from a single sitemap file, filtered to allowed prefixes."""
    xml = await crawler.fetch_xml(client, url)
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
        return [
            loc.text
            for loc in root.findall(".//sm:loc", NS)
            if loc.text and _path_allowed(loc.text)
        ]
    except ET.ParseError:
        return []


async def crawl(limit: int | None = None, dry_run: bool = False) -> None:
    output_dir = RAW_DIR / "snowflake"
    crawler = BaseCrawler(
        source="snowflake",
        output_dir=output_dir,
        delay=settings.crawl_delay_secs,
    )

    async with httpx.AsyncClient() as client:
        logger.info("fetching sitemap index: %s", SITEMAP_INDEX)
        sitemap_urls = await _parse_sitemap_index(crawler, client)
        logger.info("found %d child sitemaps", len(sitemap_urls))

        doc_urls: list[str] = []
        for sm_url in sitemap_urls:
            batch = await _parse_sitemap(crawler, client, sm_url)
            doc_urls.extend(batch)

        # deduplicate
        doc_urls = list(dict.fromkeys(doc_urls))
        logger.info("total in-scope URLs after filter: %d", len(doc_urls))

        if dry_run:
            for u in doc_urls[:20]:
                print(u)
            print(f"... ({len(doc_urls)} total)")
            return

        if limit:
            doc_urls = doc_urls[:limit]

        saved = skipped = errors = 0
        for url in doc_urls:
            if url in crawler._crawled_urls:
                skipped += 1
                continue

            html = await crawler.fetch_html(client, url)
            if not html:
                errors += 1
                continue

            extracted = crawler.extract(html, url)
            if not extracted:
                logger.warning("extraction failed: %s", url)
                errors += 1
                continue

            title, text = extracted
            if len(text.strip()) < 100:
                logger.debug("too short, skipping: %s", url)
                errors += 1
                continue

            doc_id = url_to_doc_id(url, "sf")
            doc = CrawledDoc(doc_id=doc_id, url=url, title=title, source="snowflake", text=text)
            crawler.save(doc)
            saved += 1

        logger.info("done — saved=%d skipped=%d errors=%d", saved, skipped, errors)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Crawl Snowflake documentation")
    parser.add_argument("--limit", type=int, default=None, help="cap number of pages to crawl")
    parser.add_argument("--dry-run", action="store_true", help="print URLs only, do not fetch")
    args = parser.parse_args()

    asyncio.run(crawl(limit=args.limit, dry_run=args.dry_run))
