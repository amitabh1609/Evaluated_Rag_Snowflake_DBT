"""Crawl top-200 highly-upvoted dbt Discourse threads (last 3 years).

Uses discourse.getdbt.com public API. Outputs to data/raw/dbt_discourse/.
This is the corpus differentiator — practitioner knowledge vendor docs miss.
"""

# TODO Phase 1: implement Discourse API scraper
#   - GET /top.json?period=yearly, paginate to collect 200 threads by likes
#   - For each thread: GET /t/<slug>/<id>.json, extract post bodies
#   - Combine OP + accepted/top-voted replies into one markdown doc
#   - Respect rate limits (1 req/sec); Discourse returns 429 if exceeded
