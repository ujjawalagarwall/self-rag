"""PDF text extraction helpers."""
from pathlib import Path
from document_store import extract_pdf_text


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract selectable text from a local PDF path."""
    return extract_pdf_text(Path(pdf_path).read_bytes())
