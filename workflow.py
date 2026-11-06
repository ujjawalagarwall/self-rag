"""LangGraph StateGraph orchestration for bounded self-reflective RAG."""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from nodes import (
    GraphState, retrieve, grade_documents, transform_query,
    generate, verify_generation, decide_to_generate, decide_after_verification,
)
from rag_engine import Answer

FAILURE_MESSAGE = "I couldn't verify an answer from the uploaded document. Please try a more specific question."


def create_workflow(retriever, judge, max_attempts: int = 3):
    """Compile a graph with retrieval grading, rewriting, generation and verification.

    The graph has a strict retrieval-attempt budget, preventing unbounded loops.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda state: retrieve(state, retriever))
    graph.add_node("grade_documents", lambda state: grade_documents(state, judge))
    graph.add_node("transform_query", lambda state: transform_query(state, judge))
    graph.add_node("generate", lambda state: generate(state, judge))
    graph.add_node("verify_generation", lambda state: verify_generation(state, judge))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", decide_to_generate, {
        "generate": "generate", "transform_query": "transform_query", "finish": END,
    })
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", "verify_generation")
    graph.add_conditional_edges("verify_generation", decide_after_verification, {
        "transform_query": "transform_query", "finish": END,
    })
    return graph.compile()


def run_question(workflow, question: str, max_attempts: int = 3) -> Answer:
    """Invoke the compiled graph and return a typed result for the UI/CLI."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
    result = workflow.invoke(
        {
            "question": question.strip(),
            "original_question": question.strip(),
            "documents": [],
            "generation": "",
            "attempts": 0,
            "max_attempts": max_attempts,
            "verified": False,
        },
        config={"recursion_limit": max(25, max_attempts * 8 + 5)},
    )
    verified = bool(result.get("verified"))
    return Answer(
        text=result.get("generation", "") if verified else FAILURE_MESSAGE,
        evidence=result.get("documents", []),
        attempts=result.get("attempts", 0),
        verified=verified,
    )
