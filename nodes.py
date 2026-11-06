"""LangGraph node implementations for reflective retrieval and answer verification."""
from __future__ import annotations
from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    documents: list[str]
    generation: str
    attempts: int
    max_attempts: int
    verified: bool


def retrieve(state: GraphState, retriever) -> dict:
    """Retrieve passages and count one retrieval attempt."""
    return {
        "documents": retriever.search(state["question"], limit=5),
        "attempts": state.get("attempts", 0) + 1,
    }


def grade_documents(state: GraphState, judge) -> dict:
    """Filter passages using the LLM relevance grader."""
    relevant = [doc for doc in state.get("documents", []) if judge.relevant(state["question"], doc)]
    return {"documents": relevant}


def transform_query(state: GraphState, judge) -> dict:
    """Rewrite the query to improve retrieval on the next attempt."""
    return {"question": judge.rewrite(state["question"])}


def generate(state: GraphState, judge) -> dict:
    """Generate an answer using only retained evidence."""
    return {"generation": judge.generate(state["original_question"], state["documents"])}


def verify_generation(state: GraphState, judge) -> dict:
    """Check groundedness and whether the answer addresses the original question."""
    answer = state.get("generation", "")
    docs = state.get("documents", [])
    passed = bool(answer and judge.grounded(answer, docs) and judge.useful(state["original_question"], answer))
    return {"verified": passed}


def decide_to_generate(state: GraphState) -> str:
    """Choose generation, query rewrite, or stop when retry budget is exhausted."""
    if state.get("documents"):
        return "generate"
    return "transform_query" if state["attempts"] < state["max_attempts"] else "finish"


def decide_after_verification(state: GraphState) -> str:
    if state.get("verified"):
        return "finish"
    return "transform_query" if state["attempts"] < state["max_attempts"] else "finish"
