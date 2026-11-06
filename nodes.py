"""Named steps in the reflective document QA pipeline."""
from rag_engine import Answer, answer_question


def retrieve(state: dict, retriever) -> dict:
    return {**state, "documents": retriever.search(state["question"])}


def grade_documents(state: dict, judge) -> dict:
    relevant = [
        passage for passage in state.get("documents", [])
        if judge.relevant(state["question"], passage)
    ]
    return {**state, "documents": relevant}


def transform_query(state: dict, judge) -> dict:
    return {**state, "question": judge.rewrite(state["question"])}


def generate(state: dict, judge) -> dict:
    return {**state, "generation": judge.generate(state["question"], state["documents"])}


def decide_to_generate(state: dict) -> str:
    return "generate" if state.get("documents") else "transform_query"


def grade_generation_v_documents_and_question(state: dict, judge) -> str:
    answer = state.get("generation", "")
    docs = state.get("documents", [])
    if not judge.grounded(answer, docs):
        return "not supported"
    return "useful" if judge.useful(state["question"], answer) else "not useful"
