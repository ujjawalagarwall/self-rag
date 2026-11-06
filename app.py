"""Streamlit interface for the self-reflective RAG prototype."""
from __future__ import annotations
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from document_store import DocumentRetriever, chunk_text, extract_pdf_text
from llm_checks import ReflectiveLLM
from rag_engine import answer_question

load_dotenv()
st.set_page_config(page_title="Self-RAG Document Q&A", page_icon="📄")
st.title("Self-RAG • Document Q&A")
st.caption("Retrieve, critique evidence, answer, and verify grounding in a bounded loop.")

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "replace-with-your-own-key":
    st.error("Set OPENAI_API_KEY in your local .env file to begin.")
    st.stop()

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded is None:
    st.info("Upload a document to get started.")
    st.stop()

pdf_data = uploaded.getvalue()
digest = hashlib.sha256(pdf_data).hexdigest()
if st.session_state.get("document_hash") != digest:
    try:
        extracted = extract_pdf_text(pdf_data)
        chunks = chunk_text(extracted)
        if not chunks:
            st.error("No selectable text found. Scanned/image PDFs require OCR.")
            st.stop()
        api = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with st.spinner("Indexing document..."):
            st.session_state.retriever = DocumentRetriever(api, chunks)
        st.session_state.judge = ReflectiveLLM(api)
        st.session_state.document_hash = digest
    except Exception as exc:
        st.error(f"Failed to process PDF: {type(exc).__name__}. Check your credentials and document.")
        st.stop()

question = st.text_input("Ask a question about this PDF")
if st.button("Ask", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving and verifying..."):
            response = answer_question(question, st.session_state.retriever, st.session_state.judge)
        st.subheader("Answer")
        st.write(response.text)
        st.caption(f"Attempts: {response.attempts} • LLM verification: {'passed' if response.verified else 'not passed'}")
        if response.evidence:
            with st.expander("Retrieved evidence"):
                for i, passage in enumerate(response.evidence, 1):
                    st.markdown(f"**[{i}]** {passage}")
    except Exception as exc:
        st.error(f"Could not answer: {type(exc).__name__}. Please check your API configuration.")
