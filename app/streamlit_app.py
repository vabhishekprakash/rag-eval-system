import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import html
import streamlit as st
from ingestion.loader import load_documents
from ingestion.chunker import get_chunks
from ingestion.embedder import build_vectorstore, load_vectorstore
from retrieval.retriever import retrieve_context
from retrieval.generator import generate_answer

st.set_page_config(
    page_title="RAG.eval",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Initialise session state ──────────────────────────────
# Bug fix: preserve results across sidebar interactions
if "result" not in st.session_state:
    st.session_state.result = None
if "context_docs" not in st.session_state:
    st.session_state.context_docs = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #000000;
    color: #FFFFFF;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #080808;
    border-right: 1px solid #1C1C1C;
    padding-top: 0;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * {
    color: #999 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFF !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 2.5rem 3.5rem;
    max-width: 940px;
}

/* ── Header ── */
.app-wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #FFFFFF;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.app-wordmark span {
    color: #555;
}
.app-descriptor {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #333;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

/* ── Text input ── */
.stTextInput > label { display: none; }
.stTextInput input {
    background-color: #0A0A0A !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1.1rem !important;
    transition: border-color 0.15s ease;
}
.stTextInput input::placeholder {
    color: #333 !important;
}
.stTextInput input:focus {
    border-color: #FFFFFF !important;
    box-shadow: none !important;
}

/* ── Button ── */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
    height: 100%;
    white-space: nowrap;
}
.stButton > button:hover {
    background-color: #E0E0E0 !important;
}
.stButton > button:active {
    background-color: #C0C0C0 !important;
}

/* ── Divider ── */
hr {
    border-color: #1C1C1C !important;
    margin: 1.5rem 0 !important;
}

/* ── Answer block ── */
.answer-block {
    border-top: 1px solid #FFFFFF;
    border-bottom: 1px solid #1C1C1C;
    padding: 1.5rem 0;
    margin: 1.75rem 0 1.25rem;
    font-size: 1rem;
    line-height: 1.75;
    color: #E8E8E8;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Source tags ── */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 1.75rem;
}
.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #111;
    border: 1px solid #222;
    border-radius: 2px;
    padding: 4px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #AAA;
}
.source-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #FFF;
    flex-shrink: 0;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background-color: #1C1C1C;
    border: 1px solid #1C1C1C;
    border-radius: 2px;
    margin: 0 0 2rem;
    overflow: hidden;
}
.metric-card {
    background-color: #000;
    padding: 1.25rem 1.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-label {
    font-size: 0.68rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Chunk viewer ── */
.chunk-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: #1C1C1C;
    border: 1px solid #1C1C1C;
    border-radius: 2px;
    overflow: hidden;
}
.chunk-item {
    background-color: #000;
    padding: 1rem 1.25rem;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
}
.chunk-index {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.chunk-meta-detail {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #333;
}
.chunk-text {
    font-size: 0.85rem;
    color: #555;
    line-height: 1.65;
}

/* ── Sidebar overrides ── */
.sidebar-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #333 !important;
    margin-bottom: 0.75rem;
}
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stSlider > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #555 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #0A0A0A !important;
    border-color: #222 !important;
    border-radius: 2px !important;
    color: #FFF !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #FFF !important;
}

/* ── Eval table in sidebar ── */
.eval-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    margin-top: 0.5rem;
}
.eval-table th {
    color: #333 !important;
    text-align: left;
    padding: 4px 0;
    border-bottom: 1px solid #1C1C1C;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.eval-table td {
    color: #777 !important;
    padding: 5px 0;
    border-bottom: 1px solid #111;
}
.eval-table td:not(:first-child) {
    color: #FFF !important;
    text-align: right;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #444 !important;
    background-color: #000 !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
}
.streamlit-expanderContent {
    background-color: #000 !important;
    border: 1px solid #1C1C1C !important;
    border-top: none !important;
    border-radius: 0 0 2px 2px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-label">Configuration</div>', unsafe_allow_html=True)
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
        disabled=(chunk_strategy == "semantic"),
        help="Token count per chunk. Disabled for semantic chunking."
    )

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=8, value=3
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Evaluation Results</div>', unsafe_allow_html=True)
    st.markdown("""
<table class="eval-table">
<tr><th>Strategy</th><th>Faith.</th><th>Relev.</th></tr>
<tr><td>Fixed 256</td><td>1.000</td><td>0.897</td></tr>
<tr><td>Fixed 1024</td><td>1.000</td><td>0.976</td></tr>
<tr><td>Semantic</td><td>1.000</td><td>0.976</td></tr>
</table>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#333;line-height:1.9;">'
        'Vallamalla Abhishek Prakash<br>'
        '<a href="https://github.com/vabhishekprakash" style="color:#666;text-decoration:none;">github</a>'
        ' <span style="color:#222">·</span> '
        '<a href="https://linkedin.com/in/vabhishekprakash" style="color:#666;text-decoration:none;">linkedin</a>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="app-wordmark">RAG<span>.</span>eval</div>
<div class="app-descriptor">Retrieval-Augmented Generation · Evaluation System · Local LLM</div>
""", unsafe_allow_html=True)


# ── Vectorstore loader ────────────────────────────────────
# Bug fix: only pass chunk_size for fixed strategy
@st.cache_resource(show_spinner="Initialising vector store …")
def get_vectorstore(strategy: str, size: int):
    path = f"vectorstore_{strategy}_{size}/" if strategy == "fixed" else f"vectorstore_{strategy}/"
    if os.path.exists(path):
        return load_vectorstore(save_path=path)
    docs = load_documents("data/raw")
    # Bug fix: semantic chunking doesn't accept chunk_size
    chunks = get_chunks(docs, strategy=strategy, chunk_size=size) if strategy == "fixed" \
             else get_chunks(docs, strategy=strategy)
    return build_vectorstore(chunks, save_path=path)

vectorstore = get_vectorstore(chunk_strategy, chunk_size)


# ── Query input ───────────────────────────────────────────
# Bug fix: stable key so sidebar interactions don't reset the field
query = st.text_input(
    "query",
    placeholder="Ask anything about your documents …",
    label_visibility="collapsed",
    key="query_input"
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    search = st.button("Search →", type="primary", use_container_width=True)


# ── Run retrieval on button click ─────────────────────────
if search and query:
    with st.spinner(""):
        docs = retrieve_context(query, vectorstore, k=top_k)
        res = generate_answer(query, docs)
    # Persist to session state so results survive sidebar interactions
    st.session_state.result = res
    st.session_state.context_docs = docs
    st.session_state.last_query = query


# ── Render results ────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    context_docs = st.session_state.context_docs

    # Bug fix: escape HTML in the answer to prevent XSS
    safe_answer = html.escape(result["answer"])

    st.markdown(f'<div class="answer-block">{safe_answer}</div>', unsafe_allow_html=True)

    # Sources
    tags_html = "".join(
        f'<span class="source-tag"><span class="source-dot"></span>{html.escape(s)}</span>'
        for s in result.get("sources", [])
    )
    st.markdown(f'<div class="sources-row">{tags_html}</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{len(context_docs)}</div>
        <div class="metric-label">Chunks retrieved</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{len(result["answer"].split())}</div>
        <div class="metric-label">Words in answer</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{html.escape(chunk_strategy)}</div>
        <div class="metric-label">Strategy used</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Retrieved chunks
    with st.expander(f"View {len(context_docs)} retrieved chunks"):
        chunks_html = '<div class="chunk-list">'
        for i, doc in enumerate(context_docs):
            src = html.escape(doc.metadata.get("source_file", "unknown"))
            page = html.escape(str(doc.metadata.get("page", "—")))
            # Truncate and escape the chunk text
            excerpt = html.escape(doc.page_content[:380])
            chunks_html += f"""
<div class="chunk-item">
    <div class="chunk-header">
        <span class="chunk-index">Chunk {i + 1}</span>
        <span class="chunk-meta-detail">{src} · p.{page}</span>
    </div>
    <div class="chunk-text">{excerpt}…</div>
</div>"""
        chunks_html += "</div>"
        st.markdown(chunks_html, unsafe_allow_html=True)