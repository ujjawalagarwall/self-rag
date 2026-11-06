"""PDF extraction and in-memory Chroma retrieval."""
from __future__ import annotations
import hashlib
import os
from typing import Sequence

import chromadb
import fitz
from openai import OpenAI

MAX_PAGES = 100
MAX_CHUNKS = 500


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Expected a PDF document")
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count > MAX_PAGES:
            raise ValueError(f"PDF exceeds {MAX_PAGES} pages")
        return "\n".join(page.get_text(sort=True) for page in document)


class DocumentRetriever:
    def __init__(self, client: OpenAI, texts: Sequence[str], embedding_model: str | None = None):
        texts = [text for text in texts if text.strip()]
        if not texts:
            raise ValueError("No text chunks found")
        if len(texts) > MAX_CHUNKS:
            raise ValueError(f"Document exceeds {MAX_CHUNKS} chunks")
        self.client = client
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chroma = chromadb.EphemeralClient()
        name = "doc_" + hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        self.collection = self.chroma.create_collection(name)
        embeddings = []
        for start in range(0, len(texts), 64):
            embeddings.extend(self._embed(texts[start:start + 64]))
        self.collection.add(
            ids=[str(i) for i in range(len(texts))],
            documents=texts,
            embeddings=embeddings,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def search(self, question: str, limit: int = 5) -> list[str]:
        if limit < 1:
            raise ValueError("limit must be positive")
        results = self.collection.query(
            query_embeddings=self._embed([question]),
            n_results=min(limit, self.collection.count()),
        )
        return results["documents"][0]
