"""Recursive text splitting using LangChain."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_documents(text: str, chunk_size: int = 1100, chunk_overlap: int = 180) -> list[str]:
    """Return overlapping text chunks, keeping the API simple for Chroma."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
