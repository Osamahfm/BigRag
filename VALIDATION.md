# Executed validation — 15 October 2026

Environment: Windows, Python 3.13.6, project-local `.venv`. Core and development dependencies installed successfully. No user files in the original Mini-RAG were modified. No commit, publication or deployment was performed. A local Streamlit process is running at `http://localhost:8502` because port 8501 already had a listener.

## Final automated checks

| Command | Actual result |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | **83 passed in 18.33 seconds** |
| `.\.venv\Scripts\python.exe -m ruff check app.py config.py rag ui scripts tests` | All checks passed |
| `.\.venv\Scripts\python.exe -m compileall -q app.py config.py rag ui scripts tests` | Exit code 0 |
| `.\.venv\Scripts\python.exe -m pip check` | No broken requirements found |
| Imports: Settings, Store, Pipeline, GeminiService, evaluation, app | Successful |
| HTTP GET `http://127.0.0.1:8502/_stcore/health` | `ok` |

The test suite covers real FAISS and SQLite, synthetic PDFs with page gaps/encryption, DOCX/XLSX created in memory, text/Markdown/CSV, all splitters/modes, query rewriting, reranking fallback, corrupt indexes, config invalidation, incremental embeddings, deletion including unused candidate traces, failed reindex preservation, chat isolation, citation identity/uncited paragraph rejection, corrective/reflection budgets, safe agent arguments, metric formulas and API retry/redaction. Streamlit AppTest covers example loading, chat, follow-up, session creation/reset, Arabic question, evaluation, empty selections and missing key.

Initial test runs exposed the new-document selection state issue and a regression fixture that did not actually trigger semantic embedding. Both were corrected before this final complete run. Browser inspection exposed Streamlit deprecated-width and state/default warnings, which were corrected as well.

## Real API checks

`python -m scripts.smoke --env-file ../Mini-RAG/.env` succeeded using existing credentials without printing them. The new local `.env` reuses only the Gemini key and is gitignored; the original file was not changed.

- Embedding model: `gemini-embedding-001`, RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY tasks, 768 dimensions.
- Generation model: `gemini-3.5-flash-lite`.
- Generated a supported synthetic-document answer with a validated source citation.
- Recorded smoke retrieval: 1079.12 ms; generation: 605.80 ms; total: 1684.94 ms; reported generation tokens: 197. These are one-run measurements, not a performance guarantee.
- Result file: `data/live-smoke.json`.

A separate browser test loaded the three example documents and asked “What are the advantages of MapReduce?” using Gemini. The answer cited the real Advantages section, the evidence panel reported 1 cited / 5 retrieved, and the query persisted in the local chat. The saved screenshot is `data/ui-proof.png`. The browser tab was left available as a deliverable.

The new extractor also read the existing `../Mini-RAG/document.pdf`: **27 pages, 27 readable extraction units, zero warnings**. This read-only check did not ingest or alter that original file.

## Measured evaluation

Executed:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --backend gemini --load-examples --k 3 --data-dir data\validation-gemini --output data\gemini-evaluation.json
.\.venv\Scripts\python.exe -m scripts.evaluate --backend demo --load-examples --generation --data-dir data\validation-demo --output data\demo-evaluation.json
```

Real Gemini embeddings, five labeled questions, document relevance, K=3:

| Variant | Precision@3 | Recall@3 | MRR | Hit rate | nDCG@3 | Mean local retrieval ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 8.80 |
| BM25Plus | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 7.26 |
| Hybrid | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 14.22 |
| Hybrid + lexical reranking | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 14.89 |

These are measured results on a deliberately tiny example fixture. The equal quality scores do not demonstrate hybrid or reranking superiority. Precision is relevant hits divided by K, with missing ranks counted as misses. Latency is measured after warming query/document embedding caches and optional reranker loading, excluding embedding API/model startup; it is not end-to-end live query latency. Machine load and SQLite/OneDrive behavior affect timing. The demo additionally measured clearly labeled lexical generation proxies and citation validity. No live model-judged generation benchmark was run.

## Limitations in validation

OCR and downloaded neural reranker execution were not tested with real installed optional models; their missing-dependency paths passed. External API behavior was checked with a small synthetic smoke and example queries, not stress-tested. There was no public multi-user, load, adversarial prompt-injection or tenant-isolation audit. Follow the local-only security limitations in README and future-work entries in PROGRESS.
