"""Shared result type for the LangGraph workflow."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[str]
    attempts: int
    verified: bool
