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

import os
import re
import time
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Handle API keys ──────────────────────────────────────────────────
try:
    for key in ["GROQ_API_KEY", "PINECONE_API_KEY"]:
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
              route: str, is_grounded: bool | None):
    """
    Append a structured log entry for every query.

    Why log this? It lets you:
    1. Show a monitoring dashboard ("avg latency: 1.2s, 94% grounded")
    2. Identify which queries trigger rewrites (is_grounded=False)
    3. Talk about observability in interviews with real data

    Format: one JSON object per line (JSONL) — easy to load with pandas later.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": question,
        "latency_ms": round(latency_ms, 1),
        "docs_retrieved": docs_retrieved,
        "route": route,
        "is_grounded": is_grounded,
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
        return {
            "total_queries": len(entries),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "grounded_pct": round(100 * sum(grounded) / len(grounded), 1) if grounded else None,
        }
    except Exception:
        return {}


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsLens v2",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 EarningsLens")

    # Version badge — makes it visually clear this is v2
    st.markdown(
        '<span style="background:#1d4ed8;color:white;padding:2px 10px;'
        'border-radius:12px;font-size:12px;font-weight:600;">v2 · Pinecone Cloud</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "\n\n**Adaptive RAG for Earnings Call Analysis**\n\n"
        "Ask questions about earnings call transcripts and SEC filings. "
        "The system retrieves relevant excerpts, verifies "
        "answers are grounded, and self-corrects if needed."
    )

    st.divider()

    # ── Observability panel ──────────────────────────────────────────
    stats = load_query_stats()
    if stats:
        st.markdown("### 📈 Query metrics")
        col1, col2 = st.columns(2)
        col1.metric("Total queries", stats["total_queries"])
        col2.metric("Avg latency", f"{stats['avg_latency_ms']}ms")
        if stats.get("grounded_pct") is not None:
            st.metric("Grounded answers", f"{stats['grounded_pct']}%")
        st.divider()

    # ── Company overview ─────────────────────────────────────────────
    st.markdown("### 📁 Loaded Documents")
    companies = get_loaded_companies_v2()
    if companies:
        def extract_company_name(source):
            name = os.path.basename(source)
            name = re.sub(r'\.(txt|pdf)$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s*\([A-Z]{1,5}\)', '', name)
            name = name.replace('-', ' ').replace('_', ' ')
            name = re.sub(r'\b10[KkQq]\b', '', name)
            name = re.sub(r'\b(annual|report|filing|earnings|transcript|call)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\bQ[1-4]\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\b(19|20)\d{2}\b', '', name)
            name = re.sub(r'\.com\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name)
            return name.strip()

        seen = set()
        unique_companies = []
        for name in sorted(set(extract_company_name(c) for c in companies), key=str.lower):
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_companies.append(name)
        for name in unique_companies:
            st.markdown(f"- `{name}`")
        st.caption(f"{len(unique_companies)} companies · {len(companies)} sources loaded")
    else:
        st.caption("No documents loaded — run ingest_v2.py first")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. 🔀 **Router** — decides query type (single/compare/general)\n"
        "2. 🔍 **Retriever** — fetches from Pinecone cloud index\n"
        "3. 📋 **Grader** — filters out irrelevant chunks\n"
        "4. 💡 **Generator** — produces a cited answer\n"
        "5. 🔎 **Hallucination Check** — verifies grounding\n"
        "6. ✅ **Usefulness Check** — verifies the answer is helpful\n"
        "7. ✏️ **Rewrite** — self-corrects and retries if needed\n"
        "8. ⚖️ **Compare Mode** — side-by-side company analysis"
    )

    st.divider()

    st.markdown("### Sample questions")
    sample_questions = [
        "What did the CEO say about AI investments?",
        "How did revenue change compared to last quarter?",
        "What are the biggest risks mentioned?",
        "Compare Apple and Microsoft's AI strategy",
        "Compare revenue growth across all companies",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

# ── Initialize ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    with st.spinner("Connecting to Pinecone and loading EarningsLens v2..."):
        st.session_state.graph = build_graph_v2()

# ── Chat History ─────────────────────────────────────────────────────
st.title("📊 EarningsLens")
st.caption("v2 · Adaptive RAG · LangGraph + Groq + Pinecone Cloud")
st.markdown(
    "> 💡 **What's an earnings call?** Every quarter, public companies hop on a call with Wall Street — "
    "the CEO and CFO talk numbers, drop buzzwords like *synergy*, and analysts ask questions. "
    "It's basically a company's report card, live and unscripted. "
    "Ask me anything about them — or drop in a 10-Q PDF for deeper analysis."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if meta.get("sources"):
                with st.expander("📄 Sources cited"):
                    for source in meta["sources"]:
                        st.markdown(f"- `{source}`")
            if meta.get("graph_path"):
                with st.expander("🔀 Graph path"):
                    st.markdown(" → ".join(meta["graph_path"]))
            if meta.get("latency_ms"):
                st.caption(f"⏱ {meta['latency_ms']}ms")


def format_source(doc) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_number")
    doc_type = doc.metadata.get("doc_type", "")
    if doc_type == "pdf" and page:
        return f"{source} (p. {page})"
    return source


# ── Handle Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask about earnings calls or SEC filings...") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing documents..."):
                t_start = time.time()
                result = st.session_state.graph.invoke({
                    "question": question,
                    "original_question": question,
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
                with st.expander("📄 Sources cited"):
                    for source in sources:
                        st.markdown(f"- `{source}`")
            with st.expander("🔀 Graph path"):
                st.markdown(" → ".join(graph_path))
            st.caption(f"⏱ {round(latency_ms, 1)}ms")

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
