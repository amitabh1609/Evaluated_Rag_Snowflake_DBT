.PHONY: up down crawl crawl-snowflake crawl-dbt crawl-discourse index eval ablation eval-smoke debug ui test lint fmt help

PYTHON := python
UV := uv

# ── Infrastructure ────────────────────────────────────────────────────────────

up:
	docker compose up -d qdrant postgres langfuse
	@echo "Qdrant  → http://localhost:6333"
	@echo "Langfuse → http://localhost:3000"

down:
	docker compose down

ui:
	docker compose up -d ui
	@echo "Streamlit → http://localhost:8501"

# ── Pipeline ─────────────────────────────────────────────────────────────────

crawl:
	$(UV) run python -m erag.crawl.snowflake
	$(UV) run python -m erag.crawl.dbt_docs
	$(UV) run python -m erag.crawl.dbt_discourse

crawl-snowflake:
	$(UV) run python -m erag.crawl.snowflake $(ARGS)

crawl-dbt:
	$(UV) run python -m erag.crawl.dbt_docs $(ARGS)

crawl-discourse:
	$(UV) run python -m erag.crawl.dbt_discourse $(ARGS)

index:
	$(UV) run python -m erag.index.build_index

eval:
	$(UV) run python -m erag.eval.ragas_runner

ablation:
	$(UV) run python -m erag.eval.ablation

debug:
	$(UV) run python -m erag.retrieve.debug $(ARGS)

# ── CI smoke (used by GitHub Actions) ────────────────────────────────────────

eval-smoke:
	$(UV) run python -m erag.eval.ragas_runner --smoke

# ── Dev tooling ───────────────────────────────────────────────────────────────

install:
	$(UV) sync --all-extras

test:
	$(UV) run pytest tests/ -v --tb=short

lint:
	$(UV) run ruff check src/ tests/

fmt:
	$(UV) run ruff format src/ tests/

help:
	@echo "Targets:"
	@echo "  up          Start Qdrant + Langfuse + Postgres"
	@echo "  down        Stop all containers"
	@echo "  ui          Start Streamlit UI container"
	@echo "  crawl       Crawl all three sources"
	@echo "  index       Chunk, embed, and upsert into Qdrant"
	@echo "  eval        Run full RAGAS evaluation on benchmark.yaml"
	@echo "  ablation    Run 3-config ablation and write docs/ablation_results.md"
	@echo "  eval-smoke  10-question CI smoke subset"
	@echo "  install     Install Python deps via uv"
	@echo "  test        Run pytest"
	@echo "  lint        Ruff lint"
	@echo "  fmt         Ruff format"
