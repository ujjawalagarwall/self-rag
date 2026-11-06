# Self-RAG: Reflective Document Question Answering

A Python + Streamlit + LangGraph document-question-answering app that retrieves relevant PDF passages, checks retrieval relevance, generates grounded answers, and evaluates whether an answer is supported by the source text. It can rewrite questions and retry retrieval when evidence is insufficient.

## Architecture

1. Extract and chunk text from an uploaded PDF.
2. Embed and index chunks with Chroma.
3. Retrieve the top-k passages for a question.
4. Use an LLM relevance check to retain useful passages.
5. Rewrite the question if passages are missing or irrelevant.
6. Generate an answer from context and check grounding and usefulness.
7. Display the answer with retrieved evidence or an honest insufficient-evidence response.

## Setup

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Provide your own OpenAI API key in your local `.env`, then run:

```bash
streamlit run app.py
```

The first query requires internet access for API requests; OpenAI usage may incur charges. This project has not been deployed or externally benchmarked.

## Project structure

- `app.py`: Streamlit interface
- `main.py`: command-line entry point
- `document_processing.py`: LangChain recursive text chunking
- `pdf_utils.py`: PDF text extraction
- `vectorstore.py`: vector retrieval adapter
- `nodes.py`: named retrieval and reflection steps
- `prompts.py`: prompt definitions
- `workflow.py`: LangGraph StateGraph construction and execution
- `rag_engine.py`: independent bounded-loop implementation and typed Answer result
- `document_store.py`: PDF processing and Chroma indexing
- `llm_checks.py`: LLM-based relevance, grounding, and usefulness checks
- `tests/test_rag_engine.py`: offline tests
- `.env.example`: safe example configuration; never commit a real API key

## Safety and limitations

- Uploaded documents are processed locally for extraction and then sent to OpenAI for embedding and generation; do not upload confidential PDFs without permission.
- LLM assessments are probabilistic, not formal factual guarantees.
- A LangGraph StateGraph orchestrates retrieval, grading, rewriting, generation and verification with a bounded retry budget.
- No recall, context relevance, faithfulness, or performance scores are asserted without a separate evaluation run.

