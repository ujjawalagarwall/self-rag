"""LLM checks used by the reflective retrieval graph."""
from __future__ import annotations

import json
import os
from openai import OpenAI
from prompts import (
    RELEVANCE_PROMPT,
    REWRITE_PROMPT,
    GENERATION_PROMPT,
    GROUNDING_PROMPT,
    USEFULNESS_PROMPT,
)


class ReflectiveLLM:
    def __init__(self, client: OpenAI, model: str | None = None):
        self.client = client
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def _complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _yes_no(self, prompt: str, details: str) -> bool:
        response = self._complete(prompt + " Respond with YES or NO only.", details)
        return response.strip().upper() == "YES"

    def relevant(self, question: str, passage: str) -> bool:
        return self._yes_no(RELEVANCE_PROMPT, f"Question: {question}\nPassage: {passage}")

    def rewrite(self, question: str) -> str:
        return self._complete(REWRITE_PROMPT, question) or question

    def generate(self, question: str, passages: list[str]) -> str:
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, 1))
        return self._complete(GENERATION_PROMPT, f"Question: {question}\n\nPassages:\n{context}")

    def grounded(self, answer: str, passages: list[str]) -> bool:
        return self._yes_no(
            GROUNDING_PROMPT,
            f"Passages:\n{json.dumps(passages)}\n\nAnswer:\n{answer}",
        )

    def useful(self, original_question: str, answer: str) -> bool:
        return self._yes_no(
            USEFULNESS_PROMPT,
            f"Question: {original_question}\nAnswer:\n{answer}",
        )
