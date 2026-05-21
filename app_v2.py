"""
app_v2.py — Streamlit Frontend for EarningsLens v2

What changed from v1 (app.py):
- Vector store loads from Pinecone (cloud) instead of local ChromaDB
- No local chroma_db/ directory check — Pinecone index is always available
- Ingestion check: warns if Pinecone index is empty rather than auto-running
- Query logging: every query + latency + retrieval count saved to query_log.jsonl
  (This is the observability layer — you can show this data in interviews)
- Version badge in sidebar so both apps can run side-by-side

Everything else is identical to v1 — same graph, same nodes, same UI layout.
The RAG logic (LangGraph, Groq, grading, hallucination check) is untouched.

Run with: streamlit run app_v2.py
"""

import io
import os
import re
import time
import json
import hashlib
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Handle API keys ──────────────────────────────────────────────────
try:
    for key in ["GROQ_API_KEY", "PINECONE_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # Fallback to .env

# ── Validate Pinecone key exists before anything else ────────────────
if not os.environ.get("PINECONE_API_KEY"):
    st.error(
        "**PINECONE_API_KEY not found.**\n\n"
        "- **Local:** Add it to your `.env` file\n"
        "- **Railway:** Add it under Service → Variables tab\n"
        "- **Streamlit Cloud:** Add it under App Settings → Secrets\n\n"
        "Get your key from: https://app.pinecone.io"
    )
    st.stop()

from graph_v2 import build_graph_v2
from nodes_v2 import get_loaded_companies_v2

# ── Query logging setup ──────────────────────────────────────────────
_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(_LOG_DIR, "query_log.jsonl")

def log_query(question: str, latency_ms: float, docs_retrieved: int,
              route: str, is_grounded: bool | None, grade_vector: list | None = None):
    """
    Append a structured log entry for every query.

    grade_vector: e.g. [1,0,1,0,0] — 1 = relevant at that rank position.
    Used to compute MRR. Only present on the 'retrieve' route.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": question,
        "latency_ms": round(latency_ms, 1),
        "docs_retrieved": docs_retrieved,
        "route": route,
        "is_grounded": is_grounded,
        "grade_vector": grade_vector,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_query_stats() -> dict:
    """Load aggregate stats from the query log for the sidebar."""
    if not os.path.exists(LOG_FILE):
        return {}
    try:
        entries = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        if not entries:
            return {}
        latencies = [e["latency_ms"] for e in entries]
        grounded = [e["is_grounded"] for e in entries if e["is_grounded"] is not None]

        # MRR — retrieve-route queries only
        rr_scores = []
        for e in entries:
            gv = e.get("grade_vector")
            if not gv:
                continue
            for rank, grade in enumerate(gv, start=1):
                if grade == 1:
                    rr_scores.append(1 / rank)
                    break
            else:
                rr_scores.append(0)

        return {
            "total_queries": len(entries),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "grounded_pct": round(100 * sum(grounded) / len(grounded), 1) if grounded else None,
            "mrr": round(sum(rr_scores) / len(rr_scores), 3) if rr_scores else None,
            "mrr_queries": len(rr_scores),
        }
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_companies_cached():
    return get_loaded_companies_v2()


# ── Real-time PDF ingestion ──────────────────────────────────────────

_BOILERPLATE_PHRASES = [
    "edgar online",
    "united states securities and exchange commission",
    "form 10-q",
    "form 10-k",
    "table of contents",
]

def _format_table(table: list[list]) -> str:
    rows = []
    for row in table:
        cleaned = [str(cell).strip() if cell else "-" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)

def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    hits = sum(1 for phrase in _BOILERPLATE_PHRASES if phrase in lowered)
    return hits >= 2

def process_and_ingest_pdf(uploaded_file) -> tuple[int, int]:
    """
    Extract, chunk, and upsert an uploaded PDF into Pinecone in real time.

    Uses stable chunk IDs (sha256 of filename) so re-uploading the same
    file just overwrites existing vectors — no duplicates.

    Returns: (chunks_upserted, pages_extracted)
    """
    from nodes_v2 import get_vectorstore

    filename = uploaded_file.name
    documents = []

    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            if tables:
                formatted = [_format_table(t) for t in tables if t]
                table_text = "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(formatted)
            full_text = raw_text + table_text
            if len(full_text.strip()) < 100 or _is_boilerplate(full_text):
                continue
            documents.append(Document(
                page_content=full_text,
                metadata={"source": filename, "page_number": page_num, "doc_type": "pdf"},
            ))

    if not documents:
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    source_hash = hashlib.sha256(filename.encode()).hexdigest()[:12]
    ids = [f"{source_hash}-chunk-{i:04d}" for i in range(len(chunks))]

    vs = get_vectorstore()
    vs.add_documents(chunks, ids=ids)

    return len(chunks), len(documents)


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar collapse button — sidebar is always visible */
[data-testid="collapsedControl"] { display: none !important; }

/* Main content area */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    margin-left: 0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    background-color: #111130 !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    padding-top: 1.75rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    padding: 0.25rem 0.5rem !important;
    background: rgba(99,102,241,0.04) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
}
button[kind="primary"]:hover { opacity: 0.88 !important; }

/* Secondary / outline buttons */
button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    transition: all 0.15s !important;
}
button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.08) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #c7d2fe !important;
}

/* Chat messages — base */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    margin-bottom: 1rem !important;
    padding: 0.6rem 0.75rem !important;
    gap: 0.75rem !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Avatar circles */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 14px !important;
    flex-shrink: 0 !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 0 10px rgba(99,102,241,0.35) !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    box-shadow: 0 0 10px rgba(14,165,233,0.3) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b !important; }

/* Dividers */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 1rem 0 !important; }

/* Caption text */
[data-testid="stCaptionContainer"] { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.03) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="margin-bottom:1.25rem;">'
        '<span style="font-size:1.1rem;font-weight:700;letter-spacing:-0.02em;color:#f1f5f9;">EarningsLens</span>'
        '&nbsp;&nbsp;<span style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);'
        'padding:1px 8px;border-radius:20px;font-size:10px;font-weight:600;letter-spacing:0.06em;">BETA</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Upload ───────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.75rem;font-weight:600;letter-spacing:0.06em;'
        'color:#64748b;text-transform:uppercase;margin-bottom:0.5rem;">Upload Document</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        if st.button("Analyze document", use_container_width=True, type="primary"):
            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    chunks, pages = process_and_ingest_pdf(uploaded)
                    if chunks > 0:
                        st.success(f"{pages} pages · {chunks} chunks indexed")
                        get_companies_cached.clear()
                        st.rerun()
                    else:
                        st.warning("No text found. Use a text-based PDF, not a scan.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()

    # ── Metrics ──────────────────────────────────────────────────────
    stats = load_query_stats()
    if stats:
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:600;letter-spacing:0.06em;'
            'color:#64748b;text-transform:uppercase;margin-bottom:0.5rem;">Session Metrics</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Queries", stats["total_queries"])
        col2.metric("Avg latency", f"{stats['avg_latency_ms']}ms")
        col3, col4 = st.columns(2)
        if stats.get("grounded_pct") is not None:
            col3.metric("Grounded", f"{stats['grounded_pct']}%")
        if stats.get("mrr") is not None:
            col4.metric("MRR", f"{stats['mrr']}", help=f"Mean Reciprocal Rank across {stats['mrr_queries']} retrieval queries. Closer to 1.0 is better.")
        st.divider()

    if st.button("📊 RAG Performance Stats", use_container_width=True, type="secondary"):
        st.switch_page("pages/RAG_Performance.py")


# ── Initialize ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    with st.spinner("Connecting to Pinecone and loading EarningsLens v2..."):
        st.session_state.graph = build_graph_v2()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;
               color:#f1f5f9;margin:0 0 4px 0;">
        Earnings Intelligence
    </h1>
    <p style="color:#94a3b8;font-size:0.88rem;margin:0;">
        Ask anything about earnings calls and SEC filings — answers are cited, verified, and self-corrected.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Chat History ─────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if meta.get("sources"):
                with st.expander("Sources"):
                    badges = " ".join(
                        f'<span style="display:inline-block;background:rgba(99,102,241,0.1);'
                        f'border:1px solid rgba(99,102,241,0.25);color:#a5b4fc;'
                        f'padding:2px 9px;border-radius:20px;font-size:11px;margin:2px;">{s}</span>'
                        for s in meta["sources"]
                    )
                    st.markdown(badges, unsafe_allow_html=True)
            if meta.get("graph_path"):
                with st.expander("Reasoning path"):
                    steps = " › ".join(meta["graph_path"])
                    st.markdown(
                        f'<span style="font-size:0.78rem;color:#94a3b8;">{steps}</span>',
                        unsafe_allow_html=True,
                    )
            if meta.get("latency_ms"):
                st.caption(f"{meta['latency_ms']}ms")

def format_source(doc) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_number")
    doc_type = doc.metadata.get("doc_type", "")
    if doc_type == "pdf" and page:
        return f"{source} (p. {page})"
    return source


@st.cache_data(ttl=300)
def get_known_company_terms() -> set:
    """
    Build a set of company names and tickers from the ingestion manifest.
    Used by needs_company_clarification to detect company mentions in questions.
    Auto-updates as new companies are ingested — no hardcoding needed.
    """
    terms = set()
    manifest_path = os.path.join("data", "ingested_manifest.json")
    if not os.path.exists(manifest_path):
        return terms
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for entry in manifest.values():
            if isinstance(entry, dict):
                if entry.get("ticker"):
                    terms.add(entry["ticker"].lower())
                if entry.get("company"):
                    terms.add(entry["company"].lower())
    except Exception:
        pass
    return terms


def needs_company_clarification(question: str, chat_history: list) -> bool:
    """
    Returns True if the question needs a company name to answer but doesn't
    mention one and there's no chat history to resolve it from.

    Only triggers on the first message (empty chat_history). Once a company
    has been established in the conversation, rewrite_query handles pronoun
    resolution normally.
    """
    if chat_history:
        return False  # prior turns exist — rewrite_query can resolve context

    q_lower = question.lower()

    # Check against all tickers and company names currently in the manifest.
    # Also check first word of multi-word company names (e.g. "Apple" matches "Apple Inc.")
    known_terms = get_known_company_terms()
    expanded_terms = set(known_terms)
    for term in known_terms:
        first_word = term.split()[0] if term.split() else term
        if len(first_word) >= 3:
            expanded_terms.add(first_word)
    if any(term in q_lower for term in expanded_terms):
        return False

    # Questions that only make sense for a specific company
    company_specific = [
        "ceo", "cfo", "cto", "coo", "management", "executive",
        "their", "they said", "the company", "the ceo",
        "revenue", "earnings", "guidance", "outlook", "margin",
        "profit", "loss", "quarter", "q1", "q2", "q3", "q4",
        "filing", "10-k", "10-q", "annual report",
    ]
    has_specific_language = any(w in q_lower for w in company_specific)

    # General knowledge questions don't need a company
    general_starters = (
        "what is ", "what are ", "what does ", "what do ",
        "explain ", "define ", "how does ", "how do ",
        "why is ", "why are ", "what's the difference",
        "what is eps", "what is ebitda", "what is p/e",
    )
    is_general = any(q_lower.startswith(s) for s in general_starters)

    # Compare questions name their own companies
    is_compare = "compare" in q_lower or " vs " in q_lower or " versus " in q_lower

    return has_specific_language and not is_general and not is_compare


# ── Handle Input ─────────────────────────────────────────────────────
question = st.chat_input("Ask about earnings calls or SEC filings...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build history before running clarification check
    history_messages = st.session_state.messages[:-1]
    recent = history_messages[-6:]
    chat_history = [{"role": m["role"], "content": m["content"]} for m in recent]

    if needs_company_clarification(question, chat_history):
        clarification = (
            "Which company are you asking about? "
            "For example: *Apple*, *Microsoft*, *Nvidia* — or paste a ticker symbol."
        )
        with st.chat_message("assistant"):
            st.markdown(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        st.stop()

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing documents..."):
                t_start = time.time()
                result = st.session_state.graph.invoke({
                    "question": question,
                    "original_question": question,
                    "chat_history": chat_history,
                    "documents": None,
                    "answer": None,
                    "route": None,
                    "documents_relevant": None,
                    "is_grounded": None,
                    "is_useful": None,
                    "generation_count": 0,
                    "companies_compared": None,
                })
                latency_ms = (time.time() - t_start) * 1000

            answer = result.get("answer", "I couldn't find an answer. Try rephrasing your question.")
            st.markdown(answer)

            result_docs = result.get("documents") or []
            seen_sources = set()
            sources = []
            for doc in result_docs:
                label = format_source(doc)
                if label not in seen_sources:
                    seen_sources.add(label)
                    sources.append(label)

            # ── Log the query ────────────────────────────────────────
            log_query(
                question=question,
                latency_ms=latency_ms,
                docs_retrieved=len(result_docs),
                route=result.get("route", "unknown"),
                is_grounded=result.get("is_grounded"),
                grade_vector=result.get("grade_vector"),
            )

            # ── Build graph path ─────────────────────────────────────
            graph_path = ["Router"]
            if result.get("route") == "direct":
                graph_path.append("Direct Answer")
            elif result.get("route") == "compare":
                companies = result.get("companies_compared", [])
                graph_path.append(f"Compare Mode ({', '.join(companies) if companies else 'multiple companies'})")
                graph_path.append(f"Retrieved from {len(sources)} sources")
                graph_path.append("Generated Comparison")
            else:
                graph_path.append("Retrieve (Pinecone)")
                graph_path.append(f"Grade ({len(result_docs)} relevant)")
                if result.get("generation_count", 0) > 1:
                    graph_path.append(f"Rewrite (x{result['generation_count'] - 1})")
                graph_path.append("Generate")
                if result.get("is_grounded") is not None:
                    graph_path.append(f"Hallucination Check ({'✅' if result['is_grounded'] else '❌'})")
                if result.get("is_useful") is not None:
                    graph_path.append(f"Usefulness Check ({'✅' if result['is_useful'] else '❌'})")

            metadata = {
                "sources": sources,
                "graph_path": graph_path,
                "latency_ms": round(latency_ms, 1),
            }

            if sources:
                with st.expander("Sources"):
                    badges = " ".join(
                        f'<span style="display:inline-block;background:rgba(99,102,241,0.1);'
                        f'border:1px solid rgba(99,102,241,0.25);color:#a5b4fc;'
                        f'padding:2px 9px;border-radius:20px;font-size:11px;margin:2px;">{s}</span>'
                        for s in sources
                    )
                    st.markdown(badges, unsafe_allow_html=True)
            with st.expander("Reasoning path"):
                steps = " › ".join(graph_path)
                st.markdown(
                    f'<span style="font-size:0.78rem;color:#94a3b8;">{steps}</span>',
                    unsafe_allow_html=True,
                )
            st.caption(f"{round(latency_ms, 1)}ms")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata,
            })

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                answer = "⏳ Rate limit reached — please wait 30-60 seconds and try again."
            else:
                answer = f"Something went wrong: {error_msg}. Please try again."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
