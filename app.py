"""Streamlit UI for document-grounded Q&A with citations."""

from __future__ import annotations

from html import escape

import streamlit as st

from generator import generate_answer
from ingest import build_index
from retriever import retrieve
from utils import (
    DOCUMENTS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    NO_ANSWER,
    VECTOR_STORE_DIR,
    ConfigurationError,
    ensure_directories,
    load_config,
    page_label,
)

st.set_page_config(page_title="Document Q&A with Citations", page_icon="📚", layout="wide")
ensure_directories()
try:
    config = load_config()
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #16232b; --muted: #607078; --teal: #0b7773; --line: #dce6e5; --paper: #f7faf9; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #122d34; border-right: 0; }
    [data-testid="stSidebar"] * { color: #edf7f5; }
    [data-testid="stSidebar"] .stCaption { color: #b4cdca; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    p, label, .stMarkdown { font-family: 'DM Sans', sans-serif; }
    h1 { letter-spacing: -0.03em; font-size: 2.5rem; margin-bottom: .2rem; }
    h2 { letter-spacing: -0.02em; }
    .hero { border-bottom: 1px solid var(--line); padding: 1.25rem 0 1.5rem; margin-bottom: 1.5rem; }
    .eyebrow { color: var(--teal); font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .13em; text-transform: uppercase; }
    .subtitle { color: var(--muted); font-size: 1.03rem; margin: .2rem 0 0; }
    .workspace { background: white; border: 1px solid var(--line); border-radius: 12px; padding: 1.25rem 1.35rem .5rem; box-shadow: 0 8px 28px rgba(22, 53, 58, .05); }
    .source-card { background: white; border: 1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 8px; padding: 1rem 1.1rem; margin: .65rem 0; }
    .source-meta { color: var(--muted); font-size: .85rem; margin-bottom: .55rem; }
    .source-text { color: var(--ink); line-height: 1.6; }
    [data-testid="stTextArea"] > div { background: #ffffff !important; border: 1px solid #b8ccca !important; border-radius: 8px !important; }
    [data-testid="stTextArea"] textarea { background: #ffffff !important; color: #16232b !important; caret-color: #0b7773 !important; border: 0 !important; border-radius: 8px !important; }
    [data-testid="stTextArea"] textarea::placeholder { color: #7a898d !important; opacity: 1 !important; }
    [data-testid="stTextArea"] textarea:focus { border-color: #0b7773 !important; box-shadow: 0 0 0 2px rgba(11, 119, 115, .16) !important; outline: none !important; }
    [data-testid="stTextArea"] textarea::selection { background: #bfe7e1 !important; color: #16232b !important; }
    .stButton > button { border-radius: 7px; font-weight: 600; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); padding: .65rem .8rem; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">Evidence workspace</div>'
    '<h1>Document Q&amp;A</h1>'
    '<p class="subtitle">Ask precise questions. Get answers you can trace back to the source.</p></div>',
    unsafe_allow_html=True,
)

index_ready = INDEX_PATH.exists() and METADATA_PATH.exists() and (VECTOR_STORE_DIR / "manifest.json").exists()
if not index_ready:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_passages", None)

with st.sidebar:
    st.markdown("## Document set")
    st.caption("Build a searchable evidence base from your files.")
    uploads = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Build / rebuild index", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one document before indexing.")
        else:
            for upload in uploads:
                safe_name = upload.name.replace("\\", "/").split("/")[-1]
                (DOCUMENTS_DIR / safe_name).write_bytes(upload.getbuffer())
            with st.spinner("Extracting text and building embeddings..."):
                try:
                    index_result = build_index(DOCUMENTS_DIR, VECTOR_STORE_DIR, str(config["embedding_model"]))
                    st.session_state["index_status"] = index_result
                    index_ready = True
                    st.success(f"Indexed {len(index_result['files'])} document(s) and {index_result['chunks']} chunk(s).")
                    for error in index_result["errors"]:
                        st.error(error)
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
    if st.button("Clear document set", use_container_width=True):
        for path in DOCUMENTS_DIR.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()
        for path in VECTOR_STORE_DIR.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()
        st.session_state.pop("index_status", None)
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_passages", None)
        index_ready = False
        st.success("Document set and index cleared.")

    status = st.session_state.get("index_status")
    if status:
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Documents", len(status["files"]))
        metric_col_2.metric("Chunks", status["chunks"])
    elif index_ready:
        st.info("An existing index is available.")
    else:
        st.warning("Step 1: upload documents. Step 2: click Build / rebuild index.")

st.markdown('<div class="eyebrow">Question</div>', unsafe_allow_html=True)
st.subheader("Ask your document set")
st.caption("Answers are grounded only in passages retrieved from your indexed files.")
question = st.text_area("Question", placeholder="For example: What is the annual leave allowance?", height=105, label_visibility="collapsed")
if not index_ready:
    st.caption("Index your documents first to enable questions.")

if st.button("Ask Question", type="primary", disabled=not index_ready):
    if not question.strip():
        st.warning("Enter a question before asking.")
        st.stop()
    with st.spinner("Searching the document set..."):
        try:
            passages = retrieve(question.strip(), INDEX_PATH, METADATA_PATH, str(config["embedding_model"]), int(config["top_k"]), float(config["min_score"]))
            result = generate_answer(question.strip(), passages)
        except Exception as exc:
            passages = []
            result = {"answer": None, "citation_ids": [], "valid": False, "error": str(exc), "notice": None}
        st.session_state["last_result"] = result
        st.session_state["last_passages"] = passages
else:
    result = st.session_state.get("last_result")
    passages = st.session_state.get("last_passages", [])

if result:
    if result["error"]:
        st.error(result["error"])
    else:
        if result.get("notice"):
            st.info(result["notice"])
        st.subheader("Answer")
        if result["answer"] == NO_ANSWER:
            st.info(NO_ANSWER)
        else:
            st.write(result["answer"])
            st.subheader("Sources")
            cited = set(result["citation_ids"])
            shown = [passage for passage in passages if passage["citation_id"] in cited]
            if not shown:
                st.warning("The answer did not reference a retrieved source passage.")
            for passage in shown:
                source_name = escape(str(passage["source"]))
                source_text = escape(str(passage["text"])).replace("\n", "<br>")
                st.markdown(
                    f'<div class="source-card"><strong>[{passage["citation_id"]}] {source_name}</strong>'
                    f'<div class="source-meta">{page_label(passage["page"])} &nbsp; · &nbsp; similarity {passage["score"]:.3f}</div>'
                    f'<div class="source-text">{source_text}</div></div>',
                    unsafe_allow_html=True,
                )
