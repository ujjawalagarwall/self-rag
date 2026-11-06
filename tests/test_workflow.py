"""Offline tests for the LangGraph workflow used by Streamlit and CLI."""
from workflow import create_workflow, run_question


class Retriever:
    def __init__(self):
        self.queries = []

    def search(self, question, limit=5):
        self.queries.append(question)
        return ["Paris is the capital of France."]


class Judge:
    def __init__(self, relevant=True, grounded=True, useful=True):
        self.relevance = relevant
        self.grounding = grounded
        self.usefulness = useful

    def relevant(self, question, passage):
        return self.relevance

    def rewrite(self, question):
        return question + " revised"

    def generate(self, question, passages):
        return "Paris is the capital of France. [1]"

    def grounded(self, answer, passages):
        return self.grounding

    def useful(self, question, answer):
        return self.usefulness


def test_grounded_answer():
    result = run_question(create_workflow(Retriever(), Judge()), "Capital of France?")
    assert result.verified
    assert result.attempts == 1
    assert result.evidence


def test_irrelevant_documents_trigger_bounded_rewrites():
    retriever = Retriever()
    result = run_question(
        create_workflow(retriever, Judge(relevant=False), max_attempts=3),
        "Capital?",
        max_attempts=3,
    )
    assert not result.verified
    assert result.attempts == 3
    assert retriever.queries == ["Capital?", "Capital? revised", "Capital? revised revised"]


def test_unsupported_answer_is_not_returned():
    result = run_question(
        create_workflow(Retriever(), Judge(grounded=False), max_attempts=2),
        "Capital?",
        max_attempts=2,
    )
    assert not result.verified
    assert "Paris" not in result.text


def test_unhelpful_answer_is_retried():
    result = run_question(
        create_workflow(Retriever(), Judge(useful=False), max_attempts=2),
        "Capital?",
        max_attempts=2,
    )
    assert not result.verified
    assert result.attempts == 2


def test_blank_question_fails():
    try:
        run_question(create_workflow(Retriever(), Judge()), " ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError")
