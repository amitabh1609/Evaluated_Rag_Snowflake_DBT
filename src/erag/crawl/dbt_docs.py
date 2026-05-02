"""Crawl dbt official best-practices guide and dbt-core reference documentation.

Scope (brief requirement: ~500–1,000 docs):
  - /docs/build/ — models, tests, sources, snapshots, seeds, metrics
  - /best-practices/ — project structure, modularity, style guide
  - /reference/ — dbt-core CLI, project configs, resource properties
  - /docs/introduction
  - /docs/collaborate/ — model access, groups, contracts (dbt 1.5+)
  - /docs/deploy/ — deployment environments, job scheduling

Strategy: docs.getdbt.com publishes a sitemap; parse it, filter to allowed
prefixes, crawl and extract.

Usage:
    python -m erag.crawl.dbt_docs [--limit N] [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from erag.config import RAW_DIR, settings
from erag.crawl.base import BaseCrawler, CrawledDoc, url_to_doc_id

logger = logging.getLogger(__name__)

SITEMAP_URL = "https://docs.getdbt.com/sitemap.xml"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

ALLOWED_PREFIXES = (
    "/docs/introduction",
    "/docs/build/",
    "/docs/collaborate/",
    "/docs/deploy/",
    "/docs/core-versions",
    "/best-practices/",
    "/reference/",
    "/faqs/",
    "/guides/",
    "/docs/dbt-versions/",
)

# Exclude auto-generated API pages and changelog noise
EXCLUDE_PATTERNS = (
    "/reference/dbt-jinja-functions/",  # very large, low signal for RAG
    "/docs/dbt-versions/release-notes/",  # changelogs
)


def _path_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    path = urlparse(url).path
    if any(path.startswith(ex) for ex in EXCLUDE_PATTERNS):
        return False
    return any(path.startswith(p) for p in ALLOWED_PREFIXES)


async def _parse_sitemap(crawler: BaseCrawler, client: httpx.AsyncClient) -> list[str]:
    xml = await crawler.fetch_xml(client, SITEMAP_URL)
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
        # Handle both sitemap index and regular sitemap
        locs = [loc.text for loc in root.findall(".//sm:loc", NS) if loc.text]
        # If these are child sitemaps (sitemap index), fetch each one
        if root.tag.endswith("sitemapindex"):
            all_urls: list[str] = []
            for sm_url in locs:
                child_xml = await crawler.fetch_xml(client, sm_url)
                if child_xml:
                    try:
                        child = ET.fromstring(child_xml)
                        all_urls.extend(
                            loc.text
                            for loc in child.findall(".//sm:loc", NS)
                            if loc.text and _path_allowed(loc.text)
                        )
                    except ET.ParseError:
                        pass
            return all_urls
        else:
            return [u for u in locs if _path_allowed(u)]
    except ET.ParseError as e:
        logger.error("sitemap parse error: %s", e)
        return []


async def crawl(limit: int | None = None, dry_run: bool = False) -> None:
    output_dir = RAW_DIR / "dbt_docs"
    crawler = BaseCrawler(
        source="dbt_docs",
        output_dir=output_dir,
        delay=settings.crawl_delay_secs,
    )

    async with httpx.AsyncClient() as client:
        logger.info("fetching sitemap: %s", SITEMAP_URL)
        doc_urls = await _parse_sitemap(crawler, client)
        doc_urls = list(dict.fromkeys(doc_urls))
        logger.info("total in-scope URLs: %d", len(doc_urls))

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
            if len(text.strip()) < 80:
                logger.debug("too short, skipping: %s", url)
                errors += 1
                continue

            doc_id = url_to_doc_id(url, "dbt")
            doc = CrawledDoc(doc_id=doc_id, url=url, title=title, source="dbt_docs", text=text)
            crawler.save(doc)
            saved += 1

        logger.info("done — saved=%d skipped=%d errors=%d", saved, skipped, errors)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Crawl dbt documentation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(crawl(limit=args.limit, dry_run=args.dry_run))
