# Self-RAG: Reflective Document Question Answering

A Python + Streamlit document-question-answering app that retrieves relevant PDF passages, checks retrieval relevance, generates grounded answers, and evaluates whether an answer is supported by the source text. It can rewrite questions and retry retrieval when evidence is insufficient.

> **Provenance:** This is a separately written implementation of a common self-reflective RAG pattern. The architecture was studied with reference to [AnveshKumar123/self_ragg](https://github.com/AnveshKumar123/self_ragg). It is **not** a claim of authorship of that repository. Do not present unverified evaluation metrics as results from this implementation.

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

## Files

- `app.py`: Streamlit UI and session initialization
- `rag_engine.py`: retrieval and self-reflection loop
- `document_store.py`: PDF extraction, chunking, embeddings, and retrieval
- `llm_checks.py`: structured LLM assessments and grounded generation
- `tests/test_rag_engine.py`: offline tests of retry/grounding control flow
- `.env.example`: placeholder configuration; never commit your real API key

## Safety and limitations

- Uploaded documents are processed locally for extraction and then sent to OpenAI for embedding and generation; do not upload confidential PDFs without permission.
- LLM assessments are probabilistic, not formal factual guarantees.
- A small retry budget prevents infinite re-querying or regeneration.
- No recall, context relevance, faithfulness, or performance scores are asserted without a separate evaluation run.

## Attribution

The public project [AnveshKumar123/self_ragg](https://github.com/AnveshKumar123/self_ragg) was used as an architecture reference. This repository contains independently authored code rather than a verbatim copy. The original did not include a license at the time of review.
