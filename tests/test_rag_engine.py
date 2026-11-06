"""Small unit tests for the public result type."""
from rag_engine import Answer


def test_answer_is_immutable():
    answer = Answer("Paris", ["Paris is in France"], 1, True)
    assert answer.verified is True
    assert answer.attempts == 1
