from rag_engine import answer_question


class FakeRetriever:
    def __init__(self, passage="The moon has no atmosphere."):
        self.passage = passage
        self.calls = []

    def search(self, question, limit=5):
        self.calls.append(question)
        return [self.passage]


class FakeJudge:
    def __init__(self, *, relevant=True, grounded=True, useful=True):
        self.is_relevant = relevant
        self.is_grounded = grounded
        self.is_useful = useful

    def relevant(self, question, passage):
        return self.is_relevant

    def rewrite(self, question):
        return question + " refined"

    def generate(self, question, passages):
        return "The moon has no atmosphere. [1]"

    def grounded(self, answer, passages):
        return self.is_grounded

    def useful(self, original_question, answer):
        return self.is_useful


def test_successful_grounded_answer():
    result = answer_question("What about the moon?", FakeRetriever(), FakeJudge())
    assert result.verified
    assert result.attempts == 1
    assert "[1]" in result.text


def test_rewrite_is_bounded_when_no_relevant_documents():
    retriever = FakeRetriever()
    result = answer_question("question", retriever, FakeJudge(relevant=False), max_attempts=3)
    assert not result.verified
    assert len(retriever.calls) == 3
    assert retriever.calls[-1] == "question refined refined"


def test_hallucination_grader_can_reject_answer():
    result = answer_question("question", FakeRetriever(), FakeJudge(grounded=False), max_attempts=2)
    assert not result.verified
    assert result.attempts == 2


def test_empty_question_rejected():
    try:
        answer_question(" ", FakeRetriever(), FakeJudge())
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid question to fail")
