"""Text segmentation utilities for the document QA workflow."""
from document_store import chunk_text


def split_text_into_documents(text: str, chunk_size: int = 1100, chunk_overlap: int = 180) -> list[str]:
    """Split extracted PDF text into overlapping passages."""
    return chunk_text(text, size=chunk_size, overlap=chunk_overlap)
