"""High-level workflow factory.

This project uses a bounded Python control-flow loop instead of a LangGraph
dependency. See rag_engine.py for the self-reflection algorithm.
"""
from rag_engine import answer_question


def create_workflow(retriever, judge, max_attempts: int = 3):
    """Return a callable workflow for the configured retriever and judge."""
    def run(question: str):
        return answer_question(question, retriever, judge, max_attempts=max_attempts)
    return run
