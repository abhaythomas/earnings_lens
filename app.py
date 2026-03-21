"""
app.py — Streamlit Frontend for EarningsLens

A chat interface for asking questions about earnings call transcripts.
Displays the answer, sources cited, and the path the graph took
(which nodes were activated) — great for demonstrating the agentic loop.

Run with: streamlit run app.py
"""

import os
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
        "Ask questions about earnings call transcripts. "
        "The system retrieves relevant excerpts, verifies "
        "answers are grounded, and self-corrects if needed."
    )

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. 🔀 **Router** — decides if retrieval is needed\n"
        "2. 🔍 **Retriever** — fetches relevant transcript chunks\n"
        "3. 📋 **Grader** — filters out irrelevant chunks\n"
        "4. 💡 **Generator** — produces a cited answer\n"
        "5. 🔎 **Hallucination Check** — verifies grounding\n"
        "6. ✅ **Usefulness Check** — verifies the answer is helpful\n"
        "7. ✏️ **Rewrite** — self-corrects and retries if needed"
    )

    st.divider()

    st.markdown("### Sample questions")
    sample_questions = [
        "What did the CEO say about AI investments?",
        "How did revenue change compared to last quarter?",
        "What are the biggest risks mentioned?",
        "What is the company's guidance for next quarter?",
        "Compare the profit margins across companies.",
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

# ── Handle Input ─────────────────────────────────────────────────────
# Check for prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask about earnings calls...") or prefill

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Analyzing transcripts..."):
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
            })

        answer = result.get("answer", "I couldn't find an answer. Try rephrasing your question.")
        st.markdown(answer)

        # Extract metadata for display
        sources = list(set(
            doc.metadata.get("source", "unknown")
            for doc in (result.get("documents") or [])
        ))

        # Build graph path from what happened
        graph_path = ["Router"]
        if result.get("route") == "direct":
            graph_path.append("Direct Answer")
        else:
            graph_path.append("Retrieve")
            graph_path.append(f"Grade ({len(result.get('documents') or [])} relevant)")
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
