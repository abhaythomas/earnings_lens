"""
graph_v2.py — LangGraph Adaptive RAG Workflow for EarningsLens v2

What changed from graph.py:
- Imports from nodes_v2 instead of nodes
- build_graph_v2() replaces build_graph()

Everything else — GraphState, all conditional edge functions, all node
wiring and edges — is IDENTICAL to v1.
"""

from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from nodes_v2 import (
    route_question,
    retrieve,
    grade_documents,
    generate,
    direct_answer,
    check_hallucination,
    check_usefulness,
    rewrite_query,
    compare_companies,
)


# ── State Schema ─────────────────────────────────────────────────────
# Identical to v1 — the state structure is independent of the vector store.

class GraphState(TypedDict):
    question: str                          # The user's question (may be rewritten)
    original_question: str                 # The original question (never changes)
    chat_history: List[Dict[str, Any]]     # Last N turns: [{"role": "user"|"assistant", "content": "..."}]
    documents: Optional[List[Document]]    # Retrieved document chunks
    answer: Optional[str]                  # Generated answer
    route: Optional[str]                   # "retrieve", "direct", or "compare"
    documents_relevant: Optional[bool]     # Did grading find relevant docs?
    is_grounded: Optional[bool]            # Is the answer grounded in sources?
    is_useful: Optional[bool]              # Does the answer address the question?
    generation_count: int                  # How many times we've generated (loop limit)
    companies_compared: Optional[list]     # Companies in a comparison query


# ── Conditional Edge Functions ───────────────────────────────────────

def decide_route(state: dict) -> str:
    """After routing: go to retrieval, comparison, or direct answer."""
    route = state.get("route", "retrieve")
    if route == "compare":
        return "compare"
    elif route == "retrieve":
        return "retrieve"
    else:
        return "direct_answer"


def decide_after_grading(state: dict) -> str:
    """After grading documents: generate if relevant, rewrite if not."""
    if state["documents_relevant"]:
        return "generate"
    else:
        if state["generation_count"] >= 2:
            return "give_up"
        return "rewrite_query"


def decide_after_hallucination_check(state: dict) -> str:
    """After hallucination check: proceed if grounded, rewrite if not."""
    if state["is_grounded"]:
        return "check_usefulness"
    else:
        if state["generation_count"] >= 2:
            return "end"
        return "rewrite_query"


def decide_after_usefulness_check(state: dict) -> str:
    """After usefulness check: finish if useful, rewrite if not."""
    if state["is_useful"]:
        return "end"
    else:
        if state["generation_count"] >= 2:
            return "end"
        return "rewrite_query"


# ── Build the Graph ──────────────────────────────────────────────────

def build_graph_v2() -> StateGraph:
    """
    Constructs the LangGraph workflow for v2 (Pinecone backend).

    Flow:
    START → route_question
              ├─ retrieve → grade_documents
              │               ├─ generate → check_hallucination
              │               │                ├─ check_usefulness
              │               │                │      ├─ END (useful)
              │               │                │      └─ rewrite_query → retrieve (loop)
              │               │                └─ rewrite_query → retrieve (loop)
              │               └─ rewrite_query → retrieve (loop)
              ├─ compare → END
              └─ direct_answer → END
    """

    workflow = StateGraph(GraphState)

    # ── Add nodes ────────────────────────────────────────────────────
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("check_usefulness", check_usefulness)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("compare", compare_companies)

    # ── Add edges ────────────────────────────────────────────────────

    workflow.set_entry_point("route_question")

    workflow.add_conditional_edges(
        "route_question",
        decide_route,
        {
            "retrieve": "retrieve",
            "compare": "compare",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "give_up": END,
        },
    )

    workflow.add_edge("generate", "check_hallucination")

    workflow.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            "check_usefulness": "check_usefulness",
            "rewrite_query": "rewrite_query",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "check_usefulness",
        decide_after_usefulness_check,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("compare", END)

    return workflow.compile()
