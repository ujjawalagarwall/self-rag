"""PDF ingestion and per-session retrieval using OpenAI embeddings and Chroma."""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

import chromadb
import fitz
from openai import OpenAI


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text; image-only PDFs require OCR (not included)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text(sort=True) for page in document)


def chunk_text(text: str, size: int = 1100, overlap: int = 180) -> list[str]:
    """Create overlapping character chunks; no empty or duplicate tail chunk."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    normalized = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


class DocumentRetriever:
    """Ephemeral per-session Chroma collection; no hardcoded credentials."""

    def __init__(self, client: OpenAI, texts: Sequence[str], embedding_model: str | None = None):
        if not texts:
            raise ValueError("No text chunks found in the document")
        self.client = client
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chroma = chromadb.EphemeralClient()
        self.collection = self.chroma.create_collection("doc_" + hashlib.sha256(os.urandom(16)).hexdigest()[:16])
        texts = list(texts)
        vectors = self._embed(texts)
        self.collection.add(ids=[str(i) for i in range(len(texts))], documents=texts, embeddings=vectors)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def search(self, question: str, limit: int = 5) -> list[str]:
        vectors = self._embed([question])
        results = self.collection.query(query_embeddings=vectors, n_results=min(limit, self.collection.count()))
        return results.get("documents", [[]])[0]
