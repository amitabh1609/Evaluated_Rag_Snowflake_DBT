"""Shared crawler infrastructure: robots.txt, rate limiting, text extraction, manifest."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)


@dataclass
class CrawledDoc:
    doc_id: str
    url: str
    title: str
    source: str
    text: str
    crawled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def url_to_doc_id(url: str, prefix: str) -> str:
    """Derive a stable, filesystem-safe doc_id from a URL."""
    path = urlparse(url).path.strip("/").replace("/", "_").replace("-", "_")
    path = re.sub(r"[^a-z0-9_]", "", path.lower())
    if len(path) > 80:
        path = path[:60] + "_" + hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{prefix}_{path}" if path else f"{prefix}_{hashlib.md5(url.encode()).hexdigest()[:12]}"


class BaseCrawler:
    """Async crawler with robots.txt compliance, rate limiting, and manifest tracking."""

    USER_AGENT = "evaluated-rag-portfolio/0.1 (contact: choudhryamitabh@gmail.com)"

    def __init__(
        self,
        source: str,
        output_dir: Path,
        delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir.parent / "manifest.jsonl"
        self.delay = delay
        self.timeout = timeout
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        self._crawled_urls: set[str] = self._load_crawled_urls()

    def _load_crawled_urls(self) -> set[str]:
        """Load already-crawled URLs from manifest for resumability."""
        seen: set[str] = set()
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            seen.add(json.loads(line)["url"])
                        except (json.JSONDecodeError, KeyError):
                            pass
        return seen

    def _get_robots(self, url: str) -> urllib.robotparser.RobotFileParser:
        origin = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass  # if robots.txt is unreachable, assume allowed
            self._robots[origin] = rp
        return self._robots[origin]

    def is_allowed(self, url: str) -> bool:
        return self._get_robots(url).can_fetch(self.USER_AGENT, url)

    async def _rate_limit(self, domain: str) -> None:
        last = self._last_request.get(domain, 0.0)
        wait = self.delay - (time.monotonic() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request[domain] = time.monotonic()

    async def fetch_html(self, client: httpx.AsyncClient, url: str) -> str | None:
        if url in self._crawled_urls:
            logger.debug("skip (already crawled): %s", url)
            return None
        if not self.is_allowed(url):
            logger.warning("robots.txt disallows: %s", url)
            return None

        domain = urlparse(url).netloc
        await self._rate_limit(domain)

        try:
            resp = await client.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as e:
            logger.warning("HTTP %s for %s", e.response.status_code, url)
        except Exception as e:
            logger.warning("fetch error for %s: %s", url, e)
        return None

    def extract(self, html: str, url: str) -> tuple[str, str] | None:
        """Return (title, clean_markdown_text) or None if extraction fails."""
        result = trafilatura.extract(
            html,
            url=url,
            include_tables=True,
            include_links=False,
            output_format="markdown",
            with_metadata=True,
            favor_recall=True,
        )
        if not result:
            return None
        meta = trafilatura.extract(html, url=url, output_format="json", with_metadata=True)
        title = url  # fallback
        if meta:
            try:
                title = json.loads(meta).get("title") or url
            except json.JSONDecodeError:
                pass
        return title, result

    def save(self, doc: CrawledDoc) -> None:
        path = self.output_dir / f"{doc.doc_id}.md"
        path.write_text(doc.text, encoding="utf-8")
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "url": doc.url,
                        "title": doc.title,
                        "source": doc.source,
                        "crawled_at": doc.crawled_at,
                    }
                )
                + "\n"
            )
        self._crawled_urls.add(doc.url)
        logger.info("saved %s (%d chars)", doc.doc_id, len(doc.text))

    async def fetch_xml(self, client: httpx.AsyncClient, url: str) -> str | None:
        """Fetch XML (e.g. sitemaps) — same rate limiting, no robots check for sitemaps."""
        domain = urlparse(url).netloc
        await self._rate_limit(domain)
        try:
            resp = await client.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning("xml fetch error for %s: %s", url, e)
        return None

    def make_absolute(self, base_url: str, href: str) -> str:
        return urljoin(base_url, href)
