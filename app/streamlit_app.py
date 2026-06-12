import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ingestion.loader import load_documents
from ingestion.chunker import get_chunks
from ingestion.embedder import build_vectorstore, load_vectorstore
from retrieval.retriever import retrieve_context
from retrieval.generator import generate_answer

st.set_page_config(
    page_title="RAG Eval",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0A0F;
    color: #E2E8F0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0F0F1A;
    border-right: 1px solid #1E1E2E;
}
section[data-testid="stSidebar"] * {
    color: #94A3B8 !important;
}

/* Main container */
.main .block-container {
    padding: 2.5rem 3rem;
    max-width: 900px;
}

/* Header */
.rag-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.rag-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #F8FAFC;
    letter-spacing: -0.02em;
}
.rag-accent {
    color: #6366F1;
}
.rag-subtitle {
    font-size: 0.82rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Input */
.stTextInput input {
    background-color: #0F0F1A !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 6px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Button */
.stButton > button {
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Answer block */
.answer-block {
    background-color: #0F0F1A;
    border-left: 2px solid #6366F1;
    border-radius: 0 6px 6px 0;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.97rem;
    line-height: 1.7;
    color: #CBD5E1;
}

/* Source tag */
.source-tag {
    display: inline-block;
    background-color: #1E1E2E;
    border: 1px solid #2D2D3F;
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6366F1;
    margin: 4px 4px 4px 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background-color: #0F0F1A;
    border: 1px solid #1E1E2E;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #6366F1;
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* Chunk expander */
.chunk-item {
    background-color: #0F0F1A;
    border: 1px solid #1E1E2E;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 8px;
}
.chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.chunk-text {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.6;
}

/* Divider */
hr {
    border-color: #1E1E2E !important;
    margin: 1.5rem 0 !important;
}

/* Sidebar labels */
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin-bottom: 4px;
}

/* Selectbox + Slider overrides */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #0F0F1A !important;
    border-color: #1E1E2E !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <span class="rag-title">⬡ RAG<span class="rag-accent">.</span>eval</span>
</div>
<div class="rag-subtitle">Retrieval-Augmented Generation · Evaluation System · Local LLM</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")

    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["fixed", "semantic"],
        help="Fixed: splits by token count. Semantic: splits by meaning shift."
    )

    chunk_size = st.select_slider(
        "Chunk Size",
        options=[128, 256, 512, 1024],
        value=512,
        disabled=chunk_strategy == "semantic"
    )

    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 3)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Evaluation Results</div>', unsafe_allow_html=True)
    st.markdown("""
| Strategy | Faith. | Relev. |
|---|---|---|
| Fixed 256 | 1.000 | 0.897 |
| Fixed 1024 | 1.000 | 0.976 |
| Semantic | 1.000 | 0.976 |
""")
    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#334155;">'
        'Vallamalla Abhishek Prakash<br>'
        '<a href="https://github.com/vabhishekprakash" style="color:#6366F1;">GitHub</a> · '
        '<a href="https://linkedin.com/in/vabhishekprakash" style="color:#6366F1;">LinkedIn</a>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Load vectorstore ──────────────────────────────────────
@st.cache_resource(show_spinner="Initialising vector store...")
def get_vectorstore(strategy, size):
    path = f"vectorstore_{strategy}_{size}/"
    if os.path.exists(path):
        return load_vectorstore(save_path=path)
    docs = load_documents('data/raw')
    chunks = get_chunks(docs, strategy=strategy, chunk_size=size) if strategy == "fixed" else get_chunks(docs, strategy=strategy)
    return build_vectorstore(chunks, save_path=path)

vectorstore = get_vectorstore(chunk_strategy, chunk_size)

# ── Query ─────────────────────────────────────────────────
query = st.text_input(
    "",
    placeholder="Ask anything about your documents...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 6])
with col1:
    search = st.button("SEARCH →", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────
if search and query:
    with st.spinner(""):
        context_docs = retrieve_context(query, vectorstore, k=top_k)
        result = generate_answer(query, context_docs)

    # Answer
    st.markdown(f'<div class="answer-block">{result["answer"]}</div>', unsafe_allow_html=True)

    # Sources
    sources_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in result["sources"]])
    st.markdown(f'<div style="margin-bottom:1.5rem">{sources_html}</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{len(context_docs)}</div>
            <div class="metric-label">Chunks Retrieved</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(result["answer"].split())}</div>
            <div class="metric-label">Words in Answer</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{chunk_strategy}</div>
            <div class="metric-label">Strategy Used</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieved chunks
    with st.expander(f"View {len(context_docs)} retrieved chunks"):
        for i, doc in enumerate(context_docs):
            st.markdown(f"""
            <div class="chunk-item">
                <div class="chunk-meta">Chunk {i+1} · {doc.metadata.get('source_file', 'unknown')} · page {doc.metadata.get('page', '?')}</div>
                <div class="chunk-text">{doc.page_content[:350]}...</div>
            </div>
            """, unsafe_allow_html=True)