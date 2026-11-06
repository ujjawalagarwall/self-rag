"""Bounded retrieval -> relevance -> generation -> verification workflow."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class Retriever(Protocol):
    def search(self, question: str, limit: int = 5) -> list[str]: ...


class Judge(Protocol):
    def relevant(self, question: str, passage: str) -> bool: ...
    def rewrite(self, question: str) -> str: ...
    def generate(self, question: str, passages: list[str]) -> str: ...
    def grounded(self, answer: str, passages: list[str]) -> bool: ...
    def useful(self, original_question: str, answer: str) -> bool: ...


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[str]
    attempts: int
    verified: bool


def answer_question(
    question: str, retriever: Retriever, judge: Judge, *, max_attempts: int = 3, k: int = 5
) -> Answer:
    """Run a bounded self-reflective RAG loop; never report an unverified answer as verified."""
    if not question.strip():
        raise ValueError("A non-empty question is required")
    if max_attempts < 1 or k < 1:
        raise ValueError("max_attempts and k must be positive")

    query = question.strip()
    evidence: list[str] = []
    for attempt in range(1, max_attempts + 1):
        candidates = retriever.search(query, limit=k)
        evidence = [passage for passage in candidates if judge.relevant(query, passage)]
        if evidence:
            answer = judge.generate(question, evidence)
            if answer and judge.grounded(answer, evidence) and judge.useful(question, answer):
                return Answer(text=answer, evidence=evidence, attempts=attempt, verified=True)
        if attempt < max_attempts:
            query = judge.rewrite(query)
    return Answer(
        text="I couldn't verify an answer from the uploaded document. Please try a more specific question.",
        evidence=evidence,
        attempts=max_attempts,
        verified=False,
    )
