"""
app.py
───────
AI Research Assistant — Evidence-Grounded RAG
Streamlit UI with:
  - Sidebar: document upload + ingestion pipeline
  - Chat: question input + streaming Gemini answer + inline citations
  - Sources Panel: retrieved chunks with scores
  - Evaluation Log: per-query retrieval statistics
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Cards ── */
.rag-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}
.rag-card:hover { border-color: rgba(139,92,246,0.5); }

/* ── Chat messages ── */
.msg-user {
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #f1f5f9;
}
.msg-assistant {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
}
.msg-refused {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fca5a5;
}

/* ── Citation pills ── */
.citation-pill {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #c4b5fd;
    margin: 2px 3px;
}

/* ── Score badges ── */
.score-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.score-dense { background: rgba(59,130,246,0.2); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.score-bm25  { background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.score-rrf   { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.score-rerank{ background: rgba(139,92,246,0.2); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }

/* ── Header gradient ── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.hero-sub {
    color: rgba(226,232,240,0.6);
    font-size: 0.95rem;
    margin-bottom: 24px;
}

/* ── Chunk type labels ── */
.type-text   { color: #60a5fa; }
.type-image  { color: #34d399; }
.type-table  { color: #fbbf24; }

/* ── Buttons ── */
.stButton > button, button[data-testid="baseButton-secondaryFormSubmit"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover, button[data-testid="baseButton-secondaryFormSubmit"]:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5) !important;
}

/* ── Input field ── */
div[data-baseweb="input"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
}
div[data-baseweb="input"] > input, .stTextArea textarea {
    color: #ffffff !important;
    font-size: 1rem !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader { color: #c4b5fd !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialisation ───────────────────────────────────────────────

def init_session():
    if "embedder" not in st.session_state:
        with st.spinner("Loading models (first run may take ~60s)…"):
            from ingestion.embedder import Embedder
            st.session_state.embedder = Embedder()

    if "bm25" not in st.session_state:
        from retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever()
        corpus = st.session_state.embedder.get_all_chunks()
        if corpus:
            bm25.build_index(corpus)
        st.session_state.bm25 = bm25
        st.session_state.corpus = corpus

    if "reranker" not in st.session_state:
        from retrieval.reranker import Reranker
        st.session_state.reranker = Reranker(top_k=6)

    if "generator" not in st.session_state:
        from generation.llm_generator import LLMGenerator
        st.session_state.generator = LLMGenerator()

    if "context_builder" not in st.session_state:
        from generation.context_builder import ContextBuilder
        st.session_state.context_builder = ContextBuilder()

    if "hybrid_retriever" not in st.session_state:
        from retrieval.vector_store import VectorStore
        from retrieval.hybrid_retriever import HybridRetriever
        vs = VectorStore(st.session_state.embedder)
        st.session_state.hybrid_retriever = HybridRetriever(vs, st.session_state.bm25)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role, content, meta}

    if "eval_log" not in st.session_state:
        st.session_state.eval_log = []

    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []


def rebuild_bm25():
    """Rebuild BM25 index and corpus after new documents are ingested."""
    corpus = st.session_state.embedder.get_all_chunks()
    st.session_state.corpus = corpus
    bm25 = st.session_state.bm25
    if corpus:
        bm25.build_index(corpus)


# ── Ingestion Helper ───────────────────────────────────────────────────────────

def ingest_file(uploaded_file) -> dict:
    """Save uploaded file to temp, run full ingestion pipeline, return stats."""
    from ingestion.document_parser import DocumentParser
    from ingestion.table_extractor import TableExtractor
    from ingestion.image_processor import ImageProcessor
    from ingestion.chunker import Chunker

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    stats = {"file": uploaded_file.name, "elements": 0, "chunks": 0, "added": 0}

    try:
        # 1. Parse
        parser = DocumentParser()
        elements = parser.parse(tmp_path)
        stats["elements"] = len(elements)

        # 2. Extract tables
        table_extractor = TableExtractor()
        elements = table_extractor.process_all(elements)

        # 3. Process images (Gemini vision)
        img_elements = [e for e in elements if e["element_type"] == "image"]
        if img_elements:
            image_processor = ImageProcessor()
            processed_imgs = image_processor.process_all(img_elements)
            img_map = {id(e): e for e in img_elements}
            for el, proc in zip(img_elements, processed_imgs):
                el["content"] = proc["content"]

        # 4. Rename source to original filename
        for el in elements:
            el["source"] = uploaded_file.name

        # 5. Chunk
        chunker = Chunker()
        chunks = chunker.chunk(elements)
        stats["chunks"] = len(chunks)

        # 6. Embed + index
        added = st.session_state.embedder.add_chunks(chunks, show_progress=False)
        stats["added"] = added

        # 7. Rebuild BM25
        rebuild_bm25()

    finally:
        os.unlink(tmp_path)

    return stats


# ── UI: Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 AI Research Assistant")
        st.markdown("---")

        # Knowledge base stats
        count = st.session_state.embedder.count()
        sources = st.session_state.embedder.list_sources()
        st.markdown(f"**📚 Knowledge Base**")
        st.markdown(f"- **{count}** chunks indexed")
        st.markdown(f"- **{len(sources)}** documents")

        if sources:
            with st.expander("📄 Indexed Documents", expanded=False):
                for src in sources:
                    st.markdown(f"• `{src}`")

        st.markdown("---")

        # Upload section
        st.markdown("**📤 Upload Documents**")
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("⚡ Ingest Documents", use_container_width=True):
                for uf in uploaded_files:
                    with st.status(f"Ingesting `{uf.name}`…", expanded=True) as status:
                        try:
                            stats = ingest_file(uf)
                            status.update(
                                label=f"✅ `{uf.name}` — {stats['added']} new chunks",
                                state="complete",
                            )
                            st.write(
                                f"Elements: {stats['elements']} · "
                                f"Chunks: {stats['chunks']} · "
                                f"Added: {stats['added']}"
                            )
                        except Exception as e:
                            status.update(label=f"❌ Error: {e}", state="error")

        st.markdown("---")

        # Settings
        st.markdown("**⚙️ Retrieval Settings**")
        top_k_rerank = st.slider("Top-K after reranking", 3, 10, 6, key="top_k")
        if st.session_state.get("_prev_top_k") != top_k_rerank:
            st.session_state.reranker.top_k = top_k_rerank
            st.session_state["_prev_top_k"] = top_k_rerank

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.eval_log = []
            st.session_state.last_retrieved = []
            st.rerun()


# ── UI: Main ───────────────────────────────────────────────────────────────────

def render_main():
    # Header
    st.markdown('<div class="hero-title">🔬 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Evidence-grounded answers from your documents — every claim cited, hallucinations refused.</div>', unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_sources, tab_eval = st.tabs(["💬 Chat", "📄 Sources", "📊 Evaluation Log"])

    # ── Chat Tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        _render_chat()

    # ── Sources Tab ───────────────────────────────────────────────────────────
    with tab_sources:
        _render_sources()

    # ── Eval Log Tab ─────────────────────────────────────────────────────────
    with tab_eval:
        _render_eval_log()


def _render_chat():
    """Render the chat interface."""
    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">👤 <strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            meta = msg.get("meta", {})
            if meta.get("refused"):
                css_class = "msg-refused"
                icon = "⚠️"
            else:
                css_class = "msg-assistant"
                icon = "🤖"

            st.markdown(
                f'<div class="{css_class}">{icon} <strong>Assistant:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Citations
            citations = meta.get("citations", [])
            if citations:
                pills = "".join(
                    f'<span class="citation-pill">📎 {c.get("source","?")} p.{c.get("page","?")} [{c.get("chunk_id","?")[:8]}]</span>'
                    for c in citations
                )
                st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

                with st.expander("📖 View Citations", expanded=False):
                    for i, cite in enumerate(citations, 1):
                        st.markdown(
                            f"**{i}.** `{cite.get('source')}` — Page {cite.get('page')} "
                            f"(`{cite.get('chunk_id', '')[:12]}`)"
                        )
                        if cite.get("quote"):
                            st.markdown(f"> _{cite['quote']}_")

    st.markdown("---")

    # Query input
    count = st.session_state.embedder.count()
    placeholder = (
        "Ask a question about your documents…"
        if count > 0
        else "Upload and ingest documents first, then ask a question…"
    )

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            query = st.text_input(
                "Question",
                placeholder=placeholder,
                label_visibility="collapsed",
                key="query_input",
            )
        with col_btn:
            submit = st.form_submit_button("Send ➤", use_container_width=True)

    if submit and query.strip():
        _handle_query(query.strip())


def _handle_query(query: str):
    """Run full RAG pipeline and stream answer."""
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})

    corpus = st.session_state.corpus or []
    if not corpus:
        corpus = st.session_state.embedder.get_all_chunks()
        st.session_state.corpus = corpus

    t_start = time.time()

    with st.spinner("Retrieving relevant chunks…"):
        # Hybrid retrieval
        hybrid_results = st.session_state.hybrid_retriever.search(query, corpus)

        # Reranking
        reranked = st.session_state.reranker.rerank(query, hybrid_results)

        # Context building
        context, included_chunks = st.session_state.context_builder.build(reranked)

    t_retrieval = time.time() - t_start

    # Save for Sources tab
    st.session_state.last_retrieved = reranked

    # Stream answer
    full_text = ""
    answer_placeholder = st.empty()
    answer_placeholder.markdown(
        '<div class="msg-assistant">🤖 <strong>Assistant:</strong><br><em>Thinking…</em></div>',
        unsafe_allow_html=True,
    )

    t_gen_start = time.time()
    try:
        for delta in st.session_state.generator.generate_streaming(query, context):
            full_text += delta
            answer_placeholder.markdown(
                f'<div class="msg-assistant">🤖 <strong>Assistant:</strong><br>{full_text}▌</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        full_text = json.dumps({"answer": str(e), "citations": [], "refused": True, "reasoning": str(e)})

    t_gen = time.time() - t_gen_start

    # Parse final JSON response
    from generation.llm_generator import LLMGenerator
    parsed = LLMGenerator._parse_response(full_text)

    answer_placeholder.empty()

    # Add assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": parsed.get("answer", ""),
        "meta": {
            "citations": parsed.get("citations", []),
            "refused": parsed.get("refused", False),
            "reasoning": parsed.get("reasoning", ""),
        },
    })

    # Log evaluation data
    st.session_state.eval_log.append({
        "query": query,
        "n_hybrid": len(hybrid_results),
        "n_reranked": len(reranked),
        "n_included": len(included_chunks),
        "t_retrieval_s": round(t_retrieval, 2),
        "t_generation_s": round(t_gen, 2),
        "refused": parsed.get("refused", False),
        "top_chunks": [
            {
                "chunk_id": c.get("chunk_id", "")[:12],
                "source": c.get("source", ""),
                "page": c.get("page", 0),
                "dense": c.get("dense_score", 0),
                "bm25": c.get("bm25_score", 0),
                "rrf": c.get("rrf_score", 0),
                "rerank": c.get("rerank_score", 0),
            }
            for c in reranked[:5]
        ],
    })

    st.rerun()


def _render_sources():
    """Render the retrieved sources panel."""
    chunks = st.session_state.last_retrieved
    if not chunks:
        st.markdown(
            '<div class="rag-card" style="text-align:center;color:rgba(226,232,240,0.5);">'
            '📭 Ask a question to see retrieved source chunks here.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"**{len(chunks)} chunks retrieved** (after reranking)")
    st.markdown("---")

    for i, chunk in enumerate(chunks, 1):
        etype = chunk.get("element_type", "text")
        type_icon = {"text": "📝", "image": "🖼️", "table": "📊"}.get(etype, "📄")
        type_css = f"type-{etype}"

        header = (
            f"{type_icon} **Chunk {i}** · "
            f"`{chunk.get('source', 'unknown')}` · "
            f"Page {chunk.get('page', '?')} · "
            f"<span class='{type_css}'>{etype.upper()}</span>"
        )

        scores_html = (
            f'<span class="score-badge score-dense">Dense: {chunk.get("dense_score", 0):.3f}</span>'
            f'<span class="score-badge score-bm25">BM25: {chunk.get("bm25_score", 0):.3f}</span>'
            f'<span class="score-badge score-rrf">RRF: {chunk.get("rrf_score", 0):.5f}</span>'
            f'<span class="score-badge score-rerank">Rerank: {chunk.get("rerank_score", 0):.3f}</span>'
        )

        with st.expander(f"Chunk {i} · {chunk.get('source', '')} p.{chunk.get('page', '?')}", expanded=i == 1):
            st.markdown(f"**ID:** `{chunk.get('chunk_id', '')}`")
            st.markdown(scores_html, unsafe_allow_html=True)
            st.markdown("---")
            content = chunk.get("content", "")
            st.markdown(content if len(content) <= 800 else content[:800] + "…")


def _render_eval_log():
    """Render the evaluation log."""
    log = st.session_state.eval_log
    if not log:
        st.markdown(
            '<div class="rag-card" style="text-align:center;color:rgba(226,232,240,0.5);">'
            '📊 Evaluation statistics will appear here after each query.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for entry in reversed(log):
        refused_label = "❌ REFUSED" if entry["refused"] else "✅ ANSWERED"
        with st.expander(
            f"Q: {entry['query'][:80]}{'…' if len(entry['query']) > 80 else ''} — {refused_label}",
            expanded=False,
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hybrid Results", entry["n_hybrid"])
            col2.metric("After Rerank", entry["n_reranked"])
            col3.metric("Retrieval (s)", entry["t_retrieval_s"])
            col4.metric("Generation (s)", entry["t_generation_s"])

            if entry["top_chunks"]:
                st.markdown("**Top Retrieved Chunks:**")
                rows = []
                for c in entry["top_chunks"]:
                    rows.append({
                        "Chunk ID": c["chunk_id"],
                        "Source": c["source"],
                        "Page": c["page"],
                        "Dense ↑": c["dense"],
                        "BM25 ↑": c["bm25"],
                        "RRF ↑": c["rrf"],
                        "Rerank ↑": c["rerank"],
                    })
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
