"""Streamlit UI — the front door to the evaluated RAG system.

Layout:
  - Top: chat input
  - Center: answer with inline [doc_id:chunk_id] rendered as citation badges
  - Right sidebar: Sources panel (title + URL + chunk preview + scores)
  - Below answer: Latency & Cost bar chart (per-step ms + $ estimate)
  - Below that: Faithfulness score with green/yellow/red indicator
  - Toggle: Debug retrieval panel (rewritten query + full retrieval trace)
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from erag.generate.answerer import RagAnswer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluated RAG — Snowflake & dbt",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .citation-badge {
        display: inline-block;
        background: #1f77b4;
        color: white;
        font-size: 0.7em;
        padding: 1px 5px;
        border-radius: 3px;
        margin: 0 1px;
        vertical-align: super;
        font-weight: 600;
        cursor: default;
    }
    .score-row { font-size: 0.78em; color: #666; margin-top: 2px; }
    .source-title { font-weight: 600; font-size: 0.92em; }
    .faithfulness-green  { color: #22c55e; font-weight: 700; font-size: 1.1em; }
    .faithfulness-yellow { color: #f59e0b; font-weight: 700; font-size: 1.1em; }
    .faithfulness-red    { color: #ef4444; font-weight: 700; font-size: 1.1em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []  # {question, result}


# ── Helpers ────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[([^:\]\s]+):([^\]\s]+)\]")


def _render_answer_html(answer_text: str, cited_chunks: list) -> str:
    """Replace [doc_id:chunk_index] tags with styled badge spans."""
    chunk_to_num: dict[str, int] = {}
    for i, c in enumerate(cited_chunks, 1):
        chunk_to_num[c.chunk_id] = i

    def _replace(m: re.Match) -> str:
        doc_id, suffix = m.group(1), m.group(2)
        full_id = f"{doc_id}:{suffix}"
        num = chunk_to_num.get(full_id, "?")
        return f'<sup><span class="citation-badge" title="{full_id}">[{num}]</span></sup>'

    return _CITATION_RE.sub(_replace, answer_text)


def _faithfulness_html(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 0.80:
        css = "faithfulness-green"
        label = "✓ Faithful"
    elif score >= 0.60:
        css = "faithfulness-yellow"
        label = "⚠ Partial"
    else:
        css = "faithfulness-red"
        label = "✗ Low"
    return f'<span class="{css}">Faithfulness: {score:.2f} — {label}</span>'


def _score_str(chunk) -> str:
    parts = []
    if chunk.vector_score is not None:
        parts.append(f"vec={chunk.vector_score:.3f}")
    if chunk.bm25_score is not None:
        parts.append(f"bm25={chunk.bm25_score:.3f}")
    if chunk.rerank_score is not None:
        parts.append(f"rerank={chunk.rerank_score:.3f}")
    if chunk.rrf_score is not None:
        parts.append(f"rrf={chunk.rrf_score:.4f}")
    return "  ·  ".join(parts) if parts else ""


def _run_pipeline(question: str) -> "RagAnswer":
    from erag.generate.answerer import answer as rag_answer
    return rag_answer(question)


def _build_latency_chart(timings_ms, est_cost_usd: float):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    steps = ["rewrite", "embed", "retrieve", "rerank", "generate"]
    values = [getattr(timings_ms, f"{s}_ms", 0.0) for s in steps]
    colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure(
        go.Bar(
            x=steps,
            y=values,
            marker_color=colors,
            text=[f"{v:.0f}ms" for v in values],
            textposition="outside",
        )
    )
    total_ms = sum(values)
    fig.update_layout(
        title=dict(
            text=f"Step latency  ·  Total {total_ms:.0f}ms  ·  Est. cost ${est_cost_usd:.5f}",
            font_size=13,
        ),
        yaxis_title="ms",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


# ── Sidebar: Sources ───────────────────────────────────────────────────────────

def _render_sidebar(result: "RagAnswer") -> None:
    with st.sidebar:
        st.subheader("Sources")
        if not result.cited_chunks:
            st.caption("No citations parsed from answer.")
            return

        for i, chunk in enumerate(result.cited_chunks, 1):
            with st.expander(f"[{i}] {chunk.title or chunk.doc_id}", expanded=(i == 1)):
                if chunk.url:
                    st.markdown(f"[{chunk.url}]({chunk.url})")
                preview = chunk.text_preview or ""
                if len(preview) > 400:
                    preview = preview[:400] + "…"
                st.caption(preview)
                scores = _score_str(chunk)
                if scores:
                    st.markdown(f'<div class="score-row">{scores}</div>', unsafe_allow_html=True)


# ── Main answer panel ──────────────────────────────────────────────────────────

def _render_answer(question: str, result: "RagAnswer") -> None:
    st.markdown(f"**Q:** {question}")

    rendered_html = _render_answer_html(result.answer, result.cited_chunks)
    st.markdown(rendered_html, unsafe_allow_html=True)

    if result.faithfulness_score is not None:
        st.markdown(
            _faithfulness_html(result.faithfulness_score),
            unsafe_allow_html=True,
        )

    # Latency + cost chart
    if result.timings_ms:
        fig = _build_latency_chart(result.timings_ms, result.est_cost_usd)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            timings = result.timings_ms.model_dump()
            total = sum(timings.values())
            st.caption(
                f"Latency: {total:.0f}ms  ·  "
                + "  ·  ".join(f"{k}={v:.0f}ms" for k, v in timings.items() if v)
                + f"  ·  Est. cost ${result.est_cost_usd:.5f}"
            )

    # Debug retrieval expander
    with st.expander("Debug retrieval", expanded=False):
        st.write(f"**Rewritten query:** {result.rewritten_query}")
        st.write(f"**Input tokens:** {result.input_tokens}  **Output tokens:** {result.output_tokens}")
        if result.cited_chunks:
            rows = []
            for c in result.cited_chunks:
                rows.append(
                    {
                        "chunk_id": c.chunk_id,
                        "title": c.title,
                        "vec": f"{c.vector_score:.3f}" if c.vector_score is not None else "—",
                        "bm25": f"{c.bm25_score:.3f}" if c.bm25_score is not None else "—",
                        "rerank": f"{c.rerank_score:.3f}" if c.rerank_score is not None else "—",
                        "rrf": f"{c.rrf_score:.4f}" if c.rrf_score is not None else "—",
                    }
                )
            st.dataframe(rows, use_container_width=True)

        try:
            from erag.retrieve.debug import retrieve_with_trace
            trace = retrieve_with_trace(result.rewritten_query)
            st.write("**Vector hits (top 5):**")
            for r in trace.vector_hits[:5]:
                preview = textwrap.shorten(r.metadata.get("text", ""), 120)
                st.caption(f"{r.chunk_id}  score={r.score:.4f}  {preview}")
            st.write("**BM25 hits (top 5):**")
            for r in trace.bm25_hits[:5]:
                preview = textwrap.shorten(r.metadata.get("text", ""), 120)
                st.caption(f"{r.chunk_id}  score={r.score:.4f}  {preview}")
            st.write("**After RRF (top 5):**")
            for r in trace.rrf_merged[:5]:
                st.caption(f"{r.chunk_id}  rrf={r.score:.4f}")
            if trace.reranked:
                st.write("**After reranking (top 5):**")
                for r in trace.reranked[:5]:
                    st.caption(f"{r.chunk_id}  rerank={r.score:.4f}")
        except Exception as e:
            st.caption(f"(trace unavailable: {e})")


# ── App entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Evaluated RAG — Snowflake & dbt")
    st.caption("Hybrid retrieval · BGE-large embeddings · Claude Sonnet · 4-tier benchmark")

    # Render previous turns
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            _render_answer(turn["question"], turn["result"])
            _render_sidebar(turn["result"])

    # Chat input
    question = st.chat_input("Ask a question about Snowflake or dbt…")
    if not question:
        if not st.session_state.history:
            st.info(
                "Ask anything about Snowflake or dbt — e.g.:\n\n"
                "- *How does Snowflake Time Travel work for transient tables?*\n"
                "- *What is the difference between a dbt model and a source?*\n"
                "- *How do Streams and Tasks enable CDC in Snowflake?*"
            )
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = _run_pipeline(question)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

        _render_answer(question, result)

    _render_sidebar(result)
    st.session_state.history.append({"question": question, "result": result})


main()
