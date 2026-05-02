"""Streamlit UI — the front door to the evaluated RAG system.

Layout:
  - Top: chat input
  - Center: answer with inline [doc_id:chunk_id] citation badges
  - Right sidebar: Sources panel (title + URL + chunk preview + scores)
  - Below answer: Latency & Cost bar chart (per-step ms + $ estimate)
  - Below that: Faithfulness score with green/yellow/red indicator
  - Toggle: Debug retrieval panel (rewritten query, vector hits, BM25 hits,
    RRF ranking, post-rerank ranking — all with scores and source URLs)
"""

import streamlit as st

st.set_page_config(
    page_title="Evaluated RAG — Snowflake & dbt",
    page_icon="🔍",
    layout="wide",
)

st.title("Evaluated RAG System for Snowflake & dbt Documentation")
st.caption("Hybrid retrieval · BGE-large embeddings · 4-tier hand-curated benchmark")

# TODO Phase 5: implement full UI
st.info("Phase 5 — UI implementation in progress.")

question = st.chat_input("Ask a question about Snowflake or dbt…")
if question:
    st.warning("Pipeline not yet wired. Implement in Phase 3–5.")
