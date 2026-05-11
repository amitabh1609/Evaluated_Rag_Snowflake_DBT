
"""Crawl top-200 dbt Discourse threads by likes (last 3 years).

Uses the public Discourse JSON API — no authentication required.
This is the corpus differentiator: practitioner knowledge (what goes wrong,
real-world workarounds) that vendor docs never contain.

API endpoints used:
  GET /top.json?period=yearly&page=N  — list top topics
  GET /t/<id>.json                    — full topic with all posts

Output format per doc:
  Title: <topic title>
  URL: https://discourse.getdbt.com/t/<slug>/<id>
  Tags: <comma-separated tags>
  OP: <original post text>
  ---
  Reply (likes=N): <post text>
  ...

Only posts with >= 2 likes are included (filters noise).
The original post is always included regardless of like count.

Usage:
    python -m erag.crawl.dbt_discourse [--limit N] [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from erag.config import RAW_DIR, settings
from erag.crawl.base import BaseCrawler, CrawledDoc

logger = logging.getLogger(__name__)

BASE_URL = "https://discourse.getdbt.com"
# Fetch top topics over "all time" — we filter by date client-side
TOP_URL = f"{BASE_URL}/top.json"
TOPIC_URL = f"{BASE_URL}/t/{{topic_id}}.json"

# Only include threads from the last 3 years
import datetime
CUTOFF_DATE = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3 * 365)

# Minimum likes on a reply to include it
MIN_REPLY_LIKES = 2

# Target: top 200 threads by like count
TARGET_THREADS = 200


def _clean_html_to_text(cooked: str) -> str:
    """Strip HTML tags from Discourse 'cooked' (rendered HTML) post body."""
    # Remove code blocks first (preserve as-is in output)
    cooked = re.sub(r"<pre[^>]*>.*?</pre>", lambda m: m.group(0), cooked, flags=re.DOTALL)
    # Basic tag stripping
    text = re.sub(r"<[^>]+>", " ", cooked)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    user_agent: str,
    delay: float,
    last_request: dict[str, float],
) -> dict | None:
    domain = urlparse(url).netloc
    last = last_request.get(domain, 0.0)
    wait = delay - (time.monotonic() - last)
    if wait > 0:
        await asyncio.sleep(wait)
    last_request[domain] = time.monotonic()

    try:
        resp = await client.get(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("rate-limited by Discourse, sleeping 30s")
            await asyncio.sleep(30)
        else:
            logger.warning("HTTP %s for %s", e.response.status_code, url)
    except Exception as e:
        logger.warning("fetch error %s: %s", url, e)
    return None


async def _collect_top_topic_ids(
    client: httpx.AsyncClient,
    user_agent: str,
    delay: float,
    last_request: dict[str, float],
    target: int = TARGET_THREADS,
) -> list[dict]:
    """Paginate /top.json to collect up to `target` topic summaries."""
    topics: list[dict] = []
    page = 0
    while len(topics) < target:
        url = f"{TOP_URL}?period=all&page={page}"
        data = await _fetch_json(client, url, user_agent, delay, last_request)
        if not data:
            break
        batch = data.get("topic_list", {}).get("topics", [])
        if not batch:
            break
        for t in batch:
            # Filter by date
            created = t.get("created_at", "")
            if created:
                try:
                    dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if dt < CUTOFF_DATE:
                        continue
                except ValueError:
                    pass
            topics.append(t)
        logger.info("collected %d topics so far (page %d)", len(topics), page)
        page += 1
        if len(batch) < 30:
            break  # last page

    # Sort by like_count descending and take top N
    topics.sort(key=lambda t: t.get("like_count", 0), reverse=True)
    return topics[:target]


async def _fetch_topic(
    client: httpx.AsyncClient,
    topic_id: int,
    user_agent: str,
    delay: float,
    last_request: dict[str, float],
) -> dict | None:
    url = TOPIC_URL.format(topic_id=topic_id)
    return await _fetch_json(client, url, user_agent, delay, last_request)


def _topic_to_markdown(topic_data: dict, topic_meta: dict) -> tuple[str, str]:
    """Convert a full topic JSON response into (title, markdown_text)."""
    title = topic_data.get("title", "Untitled")
    slug = topic_data.get("slug", str(topic_data.get("id", "")))
    topic_id = topic_data.get("id", "")
    url = f"{BASE_URL}/t/{slug}/{topic_id}"
    raw_tags = topic_data.get("tags", [])

    tags = ", ".join(
      tag.get("name", str(tag)) if isinstance(tag, dict) else str(tag)
      for tag in raw_tags
    )

    posts = topic_data.get("post_stream", {}).get("posts", [])

    lines = [f"# {title}", f"", f"**URL:** {url}"]
    if tags:
        lines.append(f"**Tags:** {tags}")
    lines.append("")

    for i, post in enumerate(posts):
        raw_text = _clean_html_to_text(post.get("cooked", ""))
        if not raw_text:
            continue
        like_count = post.get("like_count", 0)
        username = post.get("username", "unknown")

        if i == 0:
            lines.append("## Original Question")
            lines.append(f"*Posted by @{username}*")
            lines.append("")
            lines.append(raw_text)
        else:
            if like_count < MIN_REPLY_LIKES:
                continue
            lines.append(f"## Reply (👍 {like_count})")
            lines.append(f"*@{username}*")
            lines.append("")
            lines.append(raw_text)
        lines.append("")

    return title, "\n".join(lines)


async def crawl(limit: int | None = None, dry_run: bool = False) -> None:
    output_dir = RAW_DIR / "dbt_discourse"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.jsonl"

    # Load already-crawled topic IDs
    crawled_ids: set[int] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("source") == "dbt_discourse":
                        tid = entry.get("metadata", {}).get("topic_id")
                        if tid:
                            crawled_ids.add(int(tid))
                except (json.JSONDecodeError, KeyError):
                    pass

    last_request: dict[str, float] = {}
    user_agent = settings.crawl_user_agent
    delay = settings.crawl_delay_secs

    async with httpx.AsyncClient() as client:
        target = min(limit or TARGET_THREADS, TARGET_THREADS)
        logger.info("collecting top %d dbt Discourse topics", target)
        topic_metas = await _collect_top_topic_ids(client, user_agent, delay, last_request, target)
        logger.info("topics to process: %d", len(topic_metas))

        if dry_run:
            for t in topic_metas[:10]:
                print(f"id={t['id']} likes={t.get('like_count',0)} title={t.get('title','')}")
            print(f"... ({len(topic_metas)} total)")
            return

        saved = skipped = errors = 0
        for meta in topic_metas:
            topic_id = meta["id"]
            if topic_id in crawled_ids:
                skipped += 1
                continue

            data = await _fetch_topic(client, topic_id, user_agent, delay, last_request)
            if not data:
                errors += 1
                continue

            title, text = _topic_to_markdown(data, meta)
            if len(text.strip()) < 100:
                errors += 1
                continue

            slug = data.get("slug", str(topic_id))
            doc_id = f"discourse_{slug}_{topic_id}"[:100]
            url = f"{BASE_URL}/t/{slug}/{topic_id}"

            path = output_dir / f"{doc_id}.md"
            path.write_text(text, encoding="utf-8")

            with open(manifest_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "url": url,
                            "title": title,
                            "source": "dbt_discourse",
                            "metadata": {
                                "topic_id": topic_id,
                                "like_count": meta.get("like_count", 0),
                                "reply_count": meta.get("posts_count", 0),
                                "tags": data.get("tags", []),
                            },
                        }
                    )
                    + "\n"
                )
            crawled_ids.add(topic_id)
            logger.info("saved discourse topic %d: %s", topic_id, title[:60])
            saved += 1

        logger.info("done — saved=%d skipped=%d errors=%d", saved, skipped, errors)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Crawl dbt Discourse top threads")
    parser.add_argument("--limit", type=int, default=None, help="max threads to fetch")
    parser.add_argument("--dry-run", action="store_true", help="list topics without saving")
    args = parser.parse_args()

    asyncio.run(crawl(limit=args.limit, dry_run=args.dry_run))
