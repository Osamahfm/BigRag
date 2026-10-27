# Professional workspace upgrade

Atlas has been extended beyond the initial functional RAG implementation. The existing document library, API configuration and conversations are preserved. The application remains local at `http://localhost:8502`; restart with the same Streamlit command.

## What changed

- Six focused pages replace the crowded all-in-one sidebar: Workspace, Documents, Search studio, Evaluation, Activity and Settings.
- A consistent navy/teal design system, compact navigation, onboarding cards, clearer metrics, stage status, source cards and responsive layout.
- Background ingestion with a single worker, at most ten outstanding uploads, persistent queued/running/completed/failed/cancelled/interrupted statuses and queued-job cancellation. Workers never call Streamlit UI APIs. Running jobs are not cancellable; restart interruption is explicit and requires resubmission.
- Document table, filename/format filters, full extracted-unit preview, child/parent explorer, original download, settings-drift notice, deliberate removal action and cached-embedding coverage. Selected documents can be embedded in advance with Prepare semantic index.
- Weighted reciprocal-rank fusion with configurable semantic/keyword balance and both component scores in traces.
- Maximal marginal relevance context selection reduces duplicate evidence. It uses cached compatible semantic vectors when available, otherwise local TF-IDF. Original reciprocal-rank relevance is balanced against pairwise cosine redundancy. A per-document chunk cap is optional. This step makes no additional embedding API calls and is bypassed by Naive RAG.
- Query progress stages, stage timing chart, context budget diagnostics, configuration snapshot and trace ID. Stored candidate excerpts are capped at 1000 characters and redundant parent text is omitted; selected evidence remains complete within the context budget.
- Search studio compares methods before generation, shows ranking/context changes and exports experiments. Changed settings, selection or corpus invalidate the displayed experiment.
- Persistent engine and validated model/retrieval/chunk settings; conversation-specific document selection, modes and rewrite/expansion/routing controls.
- Helpful/Needs review feedback, review notes, saved answers and Markdown answer downloads. Feedback references message IDs and cascades when a conversation is cleared. It is human feedback, not an automatic quality score.
- Evaluation history persists run labels, exact settings, dataset fingerprint, library revision and per-question results. Historical runs can be compared; different datasets/K/corpora are clearly identified. Retrieval metrics evaluate candidate rankings before MMR; Search studio shows context diversity separately.
- Activity dashboard shows measured median/p95 latency, API attempts, ingestion/events and saved answers. These are observed local metrics, not guarantees.
- Stronger citation validation prevents Markdown headings from hiding uncited body paragraphs. Evidence highlights are HTML-escaped, with RTL display for Arabic excerpts.
- Additive SQLite tables/indexes preserve the existing library. Chunk ordinals preserve reading order; older chunks use parent/page metadata and text offsets as a fallback.

## Changed / added files

Added: `rag/diversity.py`, `rag/jobs.py`, `ui/theme.py`, `ui/application.py`, `ui/workspace.py`, `tests/test_professional.py`, this report.

Updated: `app.py`, `config.py`, `rag/chunking.py`, `rag/storage.py`, `rag/retrieval.py`, `rag/pipeline.py`, `rag/generation.py`, `rag/evaluation.py`, `ui/components.py`, `.streamlit/config.toml`, UI tests and project documentation.

## Verification

The expanded complete suite passed **103 tests**, with Ruff, syntax/import and pip checks passing. Targeted UI tests were rerun after final interface refinements. A live Gemini smoke using the requested embedding and generation models succeeded with a valid citation; its measured result is in `data/professional-live-smoke.json`. Browser checks and screenshots are recorded in VALIDATION.md.

## Practical limits

This remains a local single-user application. The queue is a bounded in-process worker, not a distributed durable execution engine; only job history is durable. Semantic rank weighting and MMR are tunable heuristics, not learned or calibrated confidence estimates. Real OCR and neural reranker execution still require the optional dependencies/models. Live token streaming remains deferred so citations are validated before display. Public multi-user hardening, full-document summarization and larger human-labeled evaluation remain future work.
