"""Command-line entry point for document question answering."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from document_processing import split_text_into_documents
from llm_checks import ReflectiveLLM
from pdf_utils import extract_text_from_pdf
from workflow import create_workflow
from vectorstore import create_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question about a PDF")
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument("question", help="Question to ask about the document")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in a local .env file")
    text = extract_text_from_pdf(args.pdf)
    chunks = split_text_into_documents(text)
    if not chunks:
        parser.error("PDF contains no selectable text")
    client = OpenAI()
    retriever = create_vectorstore(chunks, client)
    result = create_workflow(retriever, ReflectiveLLM(client))(args.question)
    print(result.text)


if __name__ == "__main__":
    main()
