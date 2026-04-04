"""
app.py — Streamlit Frontend for EarningsLens

A chat interface for asking questions about earnings call transcripts.
Displays the answer, sources cited, and the path the graph took
(which nodes were activated) — great for demonstrating the agentic loop.

Run with: streamlit run app.py
"""

import os
import re
import streamlit as st

# ── Handle API key from Streamlit Cloud secrets or .env ──────────────
# Streamlit Cloud uses st.secrets; local dev uses .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # Fallback to .env (handled by python-dotenv in nodes.py)

# ── Auto-ingest if vector store doesn't exist ────────────────────────
if not os.path.exists("chroma_db"):
    from ingest import main as run_ingest
    with st.spinner("First run — building vector store from transcripts..."):
        run_ingest()

from graph import build_graph
from nodes import get_loaded_companies

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsLens",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 EarningsLens")
    st.markdown(
        "**Adaptive RAG for Earnings Call Analysis**\n\n"
        "Ask questions about earnings call transcripts and SEC filings. "
        "The system retrieves relevant excerpts, verifies "
        "answers are grounded, and self-corrects if needed."
    )

    st.divider()

    # Company overview — shows what's loaded
    st.markdown("### 📁 Loaded Documents")
    companies = get_loaded_companies()
    if companies:
        def extract_company_name(source):
            name = os.path.basename(source)
            # Strip extension
            name = re.sub(r'\.(txt|pdf)$', '', name, flags=re.IGNORECASE)
            # Remove parenthesized tickers first e.g. "(AAPL)", "(MSFT)"
            name = re.sub(r'\s*\([A-Z]{1,5}\)', '', name)
            # Normalize separators (hyphens/underscores → spaces)
            name = name.replace('-', ' ').replace('_', ' ')
            # Remove common SEC/filing prefixes and tokens
            # e.g. "10Q Q2 2025 Apple" → "Apple"
            name = re.sub(r'\b10[KkQq]\b', '', name)
            name = re.sub(r'\b(annual|report|filing|earnings|transcript|call)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\bQ[1-4]\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\b(19|20)\d{2}\b', '', name)  # Remove years
            name = re.sub(r'\.com\b', '', name, flags=re.IGNORECASE)
            # Collapse multiple spaces
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
        st.caption("No documents loaded yet")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. 🔀 **Router** — decides query type (single/compare/general)\n"
        "2. 🔍 **Retriever** — fetches relevant transcript chunks\n"
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
    with st.spinner("Loading EarningsLens..."):
        st.session_state.graph = build_graph()

# ── Chat History ─────────────────────────────────────────────────────
st.title("📊 EarningsLens")
st.caption("Adaptive RAG for Earnings Call Analysis — powered by LangGraph + Groq")
st.markdown(
    "> 💡 **What's an earnings call?** Every quarter, public companies hop on a call with Wall Street — "
    "the CEO and CFO talk numbers, drop buzzwords like *synergy*, and analysts ask questions. "
    "It's basically a company's report card, live and unscripted. "
    "Ask me anything about them — or drop in a 10-Q PDF for deeper analysis."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources and graph path for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if meta.get("sources"):
                with st.expander("📄 Sources cited"):
                    for source in meta["sources"]:
                        st.markdown(f"- `{source}`")
            if meta.get("graph_path"):
                with st.expander("🔀 Graph path"):
                    st.markdown(" → ".join(meta["graph_path"]))

# ── Helper: Format source citation ───────────────────────────────────
def format_source(doc) -> str:
    """
    Returns a human-readable source string.
    For PDFs, includes the page number: 'apple_10q.pdf (p. 4)'
    For TXTs, just the filename: 'apple_q4_2024.txt'
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_number")
    doc_type = doc.metadata.get("doc_type", "")

    if doc_type == "pdf" and page:
        return f"{source} (p. {page})"
    return source

# ── Handle Input ─────────────────────────────────────────────────────
# Check for prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask about earnings calls or SEC filings...") or prefill

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the graph
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing documents..."):
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

            answer = result.get("answer", "I couldn't find an answer. Try rephrasing your question.")
            st.markdown(answer)

            # ── Build source citations with page numbers for PDFs ──
            result_docs = result.get("documents") or []
            seen_sources = set()
            sources = []
            for doc in result_docs:
                label = format_source(doc)
                if label not in seen_sources:
                    seen_sources.add(label)
                    sources.append(label)

            # Build graph path from what happened
            graph_path = ["Router"]
            if result.get("route") == "direct":
                graph_path.append("Direct Answer")
            elif result.get("route") == "compare":
                companies = result.get("companies_compared", [])
                graph_path.append(f"Compare Mode ({', '.join(companies) if companies else 'multiple companies'})")
                graph_path.append(f"Retrieved from {len(sources)} sources")
                graph_path.append("Generated Comparison")
            else:
                graph_path.append("Retrieve")
                graph_path.append(f"Grade ({len(result_docs)} relevant)")
                if result.get("generation_count", 0) > 1:
                    graph_path.append(f"Rewrite (x{result['generation_count'] - 1})")
                graph_path.append("Generate")
                if result.get("is_grounded") is not None:
                    graph_path.append(f"Hallucination Check ({'✅' if result['is_grounded'] else '❌'})")
                if result.get("is_useful") is not None:
                    graph_path.append(f"Usefulness Check ({'✅' if result['is_useful'] else '❌'})")

            metadata = {"sources": sources, "graph_path": graph_path}

            if sources:
                with st.expander("📄 Sources cited"):
                    for source in sources:
                        st.markdown(f"- `{source}`")
            with st.expander("🔀 Graph path"):
                st.markdown(" → ".join(graph_path))

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata,
            })

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                answer = "⏳ Rate limit reached — the free Groq API allows a limited number of requests per minute. Please wait 30-60 seconds and try again."
            else:
                answer = f"Something went wrong: {error_msg}. Please try again."

            st.warning(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })
