"""Tests for chunking and PDF input validation."""
import pytest
from document_processing import split_text_into_documents
from document_store import extract_pdf_text


def test_chunking_long_text():
    chunks = split_text_into_documents("A sentence. " * 300, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunking_empty_text():
    assert split_text_into_documents("") == []


def test_invalid_pdf_rejected():
    with pytest.raises(ValueError):
        extract_pdf_text(b"not a PDF")
