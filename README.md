# Document Q&A with Citations

A Streamlit Retrieval-Augmented Generation (RAG) application for asking questions about a local set of PDF, DOCX, and TXT documents. Answers are generated only from retrieved passages and must cite those passages with IDs such as `[S1]`.

## Setup on Windows

From PowerShell:

```powershell
cd "C:\Users\LENOVO\Desktop\LLM\intern task\document_qa"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```powershell
streamlit run app.py
```

The first indexing run downloads the `all-MiniLM-L6-v2` embedding model. The LLM receives only the top retrieved passages, never the full document set. If `OPENAI_API_KEY` is not configured, the app automatically uses a local extractive answer from the retrieved passages, so the core flow still works without an API account.

## How It Works

```text
PDF/DOCX/TXT -> extraction with source metadata -> overlapping chunks
             -> normalized Sentence Transformer embeddings -> FAISS IndexFlatIP
             -> top-k retrieval and score threshold -> grounded OpenAI answer
             -> citation validation -> answer and exact source passages
```

- PDFs retain page numbers; DOCX and TXT use `Page unavailable`.
- `MIN_SCORE` prevents weak matches from being sent to the LLM; the default is `0.25` and can be raised after evaluating your document set.
- The vector store includes a manifest that verifies the embedding model, vector dimension, and metadata count before retrieval.
- Missing or weak evidence returns: `The provided documents do not contain enough information to answer this question.`
- Invalid citation IDs are rejected and logged rather than displayed as trusted evidence.
- Extraction errors are reported in the sidebar while usable files can still be indexed.
- Internal placeholder files such as `.gitkeep` are ignored during indexing.
- Use `Clear document set` in the sidebar when starting a new collection; this removes stored uploads and the active index.

Uploaded text is stored locally under `data/documents/`. When a question is answered, only the retrieved passages are sent to the configured OpenAI-compatible API. Do not upload confidential material unless that external processing is acceptable for your use case.

## Project Layout

`app.py` contains the Streamlit UI. `ingest.py` extracts, chunks, embeds, and writes the FAISS index. `retriever.py` performs thresholded search. `generator.py` builds the grounded prompt and validates citation IDs. `utils.py` contains shared paths and helpers.

Runtime files are written to `data/documents/` and `vector_store/`; both are ignored by Git. Example evaluation questions are in `tests/test_cases.json` and cover direct lookup, multi-document questions, and unavailable information.

## Evaluation

For a small manual evaluation, run each question in `tests/test_cases.json` and check:

1. The expected source appears in the top five retrieved results.
2. Each factual answer sentence includes a supplied citation ID.
3. Every displayed source passage supports the statement it cites.
4. The unavailable-information case returns the exact no-answer message.
5. Empty, corrupted, and unsupported files produce a visible error.

Do not commit `.env`, uploaded documents, or generated vector-store files.