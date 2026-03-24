"""
graph.py — LangGraph Adaptive RAG Workflow for EarningsLens

This file defines the "brain" of the system — a directed graph where:
- Each NODE is a function from nodes.py
- Each EDGE is a decision (conditional routing)

The flow:
1. Router decides: retrieve or answer directly?
2. If retrieve → fetch chunks → grade them
3. If relevant chunks found → generate answer → check hallucination → check usefulness
4. If anything fails → rewrite query and try again (max 3 attempts)

This is called "Adaptive RAG" because it self-corrects:
- Bad retrieval? → Rewrite query
- Hallucinated answer? → Rewrite query and regenerate
- Unhelpful answer? → Rewrite query and regenerate
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from nodes import (
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

# ── Define the State ─────────────────────────────────────────────────
# This is the data that flows through the graph.
# Every node can read and update this state.

class GraphState(TypedDict):
    question: str                          # The user's question (may be rewritten)
    original_question: str                 # The original question (never changes)
    documents: Optional[List[Document]]    # Retrieved document chunks
    answer: Optional[str]                  # Generated answer
    route: Optional[str]                   # "retrieve", "direct", or "compare"
    documents_relevant: Optional[bool]     # Did grading find relevant docs?
    is_grounded: Optional[bool]            # Is the answer grounded in sources?
    is_useful: Optional[bool]              # Does the answer address the question?
    generation_count: int                  # How many times we've generated (loop limit)
    companies_compared: Optional[list]     # Companies in a comparison query


# ── Conditional Edge Functions ───────────────────────────────────────
# These functions decide which node to go to next based on state.

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
        # No relevant documents — rewrite and try again
        if state["generation_count"] >= 2:
            return "give_up"  # Prevent infinite loops
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

def build_graph() -> StateGraph:
    """
    Constructs the LangGraph workflow.

    Visual representation:

    START → route_question
              ├─ retrieve → grade_documents
              │               ├─ generate → check_hallucination
              │               │                ├─ check_usefulness
              │               │                │      ├─ END (useful)
              │               │                │      └─ rewrite_query → retrieve (loop)
              │               │                └─ rewrite_query → retrieve (loop)
              │               └─ rewrite_query → retrieve (loop)
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

    # Entry point
    workflow.set_entry_point("route_question")

    # After routing: retrieve, compare, or answer directly
    workflow.add_conditional_edges(
        "route_question",
        decide_route,
        {
            "retrieve": "retrieve",
            "compare": "compare",
            "direct_answer": "direct_answer",
        },
    )

    # After retrieval: always grade documents
    workflow.add_edge("retrieve", "grade_documents")

    # After grading: generate if relevant, rewrite if not
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "give_up": END,
        },
    )

    # After generation: always check hallucination
    workflow.add_edge("generate", "check_hallucination")

    # After hallucination check: proceed or rewrite
    workflow.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            "check_usefulness": "check_usefulness",
            "rewrite_query": "rewrite_query",
            "end": END,
        },
    )

    # After usefulness check: finish or rewrite
    workflow.add_conditional_edges(
        "check_usefulness",
        decide_after_usefulness_check,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
        },
    )

    # After rewriting: always retrieve again
    workflow.add_edge("rewrite_query", "retrieve")

    # Direct answer goes straight to end
    workflow.add_edge("direct_answer", END)

    # Compare goes straight to end (it handles its own retrieval internally)
    workflow.add_edge("compare", END)

    return workflow.compile()


# ── Run standalone for testing ───────────────────────────────────────

def query(question: str) -> dict:
    """Run a single question through the graph."""
    app = build_graph()
    result = app.invoke({
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
    return result


if __name__ == "__main__":
    # Quick test
    result = query("What did the CEO say about AI investments?")
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
