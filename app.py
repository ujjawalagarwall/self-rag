"""Streamlit interface for PDF question answering."""
import hashlib
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from document_processing import split_text_into_documents
from document_store import DocumentRetriever, extract_pdf_text
from llm_checks import ReflectiveLLM
from workflow import create_workflow, run_question

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Self-RAG Document Q&A", page_icon="📄")
st.title("Self-RAG • Document Q&A")
st.caption("Ask questions grounded in your PDF, with retrieval and answer verification.")

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "replace-with-your-own-key":
    st.error("Set OPENAI_API_KEY in your local .env file.")
    st.stop()

uploaded = st.file_uploader("Upload a text-based PDF (up to 10 MB)", type=["pdf"])
if uploaded is None:
    st.info("Upload a document to get started.")
    st.stop()

pdf_data = uploaded.getvalue()
if len(pdf_data) > MAX_PDF_BYTES:
    st.error("PDF exceeds the 10 MB limit.")
    st.stop()

digest = hashlib.sha256(pdf_data).hexdigest()
if st.session_state.get("document_hash") != digest:
    try:
        extracted = extract_pdf_text(pdf_data)
        chunks = split_text_into_documents(extracted)
        if not chunks:
            st.error("No selectable text found. Scanned PDFs require OCR.")
            st.stop()
        api = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0, max_retries=2)
        with st.spinner("Indexing document..."):
            retriever = DocumentRetriever(api, chunks)
        st.session_state.retriever = retriever
        st.session_state.judge = ReflectiveLLM(api)
        st.session_state.document_hash = digest
    except Exception:
        logger.exception("Document indexing failed")
        st.error("Could not index this PDF. Check its contents and API configuration.")
        st.stop()

question = st.text_input("Ask a question about this PDF")
if st.button("Ask", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving and checking the answer..."):
            workflow = create_workflow(st.session_state.retriever, st.session_state.judge)
            response = run_question(workflow, question)
        st.subheader("Answer")
        st.write(response.text)
        st.caption(f"Retrieval attempts: {response.attempts} • LLM checks: {'passed' if response.verified else 'not passed'}")
        if response.evidence:
            with st.expander("Retrieved passages"):
                for i, passage in enumerate(response.evidence, 1):
                    st.markdown(f"**[{i}]** {passage}")
    except Exception:
        logger.exception("Answer generation failed")
        st.error("Could not complete the question. Check your API quota or try again.")
