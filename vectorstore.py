"""Vector retrieval utilities backed by Chroma and OpenAI embeddings."""
from document_store import DocumentRetriever


def create_vectorstore(documents: list[str], client):
    """Index document chunks into an ephemeral collection."""
    return DocumentRetriever(client, documents)


def get_retriever(vectorstore: DocumentRetriever) -> DocumentRetriever:
    """Return a retriever exposing search(question, limit)."""
    return vectorstore
