import sqlite3
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from config import ROOT, Settings
from rag.diversity import diversify
from rag.errors import RAGError
from rag.evaluation import compare
from rag.generation import validate_citations
from rag.jobs import IngestionJobs
from rag.library import Library
from rag.pipeline import Pipeline
from rag.retrieval import Retriever, rrf
from rag.storage import Store
from ui.components import highlighted_excerpt


def test_weighted_rrf_preserves_component_scores():
    dense = [{"id": "a", "dense_score": 0.9}, {"id": "b", "dense_score": 0.7}]
    sparse = [{"id": "b", "sparse_score": 4.2}, {"id": "a", "sparse_score": 2.1}]
    result = rrf([dense, sparse], k=60, weights=[0.8, 0.2])
    assert result[0]["id"] == "a"
    assert result[0]["score"] == pytest.approx(0.8/61+0.2/62)
    assert result[0]["dense_score"] == 0.9 and result[0]["sparse_score"] == 2.1
    assert rrf([dense, sparse], weights=[0, 1])[0]["id"] == "b"


@pytest.mark.parametrize("weights", [[1], [-1, 2], [0, 0], [float("nan"), 1]])
def test_invalid_fusion_weights(weights):
    with pytest.raises(ValueError):
        rrf([[], []], weights=weights)


def test_keyword_weight_does_not_call_gemini(corpus, settings, service):
    store, _ = corpus
    retriever = Retriever(store, replace(settings, dense_weight=0), service)
    assert retriever.search("MapReduce", "hybrid")
    service.embed.assert_not_called()


def test_mmr_avoids_duplicates_without_api(store, settings, service):
    candidates = [{"id": "a", "document_id": "first", "text": "MapReduce scalable distributed cluster"},
        {"id": "b", "document_id": "first", "text": "MapReduce scalable distributed cluster"},
        {"id": "c", "document_id": "second", "text": "Spark memory iterative workloads"}]
    diverse, method = diversify(candidates, store, replace(settings, mmr_lambda=0.2))
    assert [c["id"] for c in diverse[:2]] == ["a", "c"]
    assert method == "local TF-IDF cosine"
    service.embed.assert_not_called()
    capped, _ = diversify(candidates, store, replace(settings, max_chunks_per_document=1))
    assert len(capped) == 2
    assert diversify(candidates, store, replace(settings, diversity=False))[0] == candidates


def test_mmr_uses_existing_semantic_vectors(corpus, settings, service):
    store, _ = corpus
    retriever = Retriever(store, settings, service)
    candidates = retriever.search("MapReduce", "dense")
    calls = service.embed.call_count
    _, method = diversify(candidates, store, settings)
    assert method == "cached embedding cosine" and service.embed.call_count == calls


def test_pipeline_emits_stages_and_reproducible_trace(corpus, settings, service):
    store, _ = corpus
    progress = Mock()
    answer = Pipeline(store, settings, service).ask("MapReduce", progress=progress)
    stages = [call.args[0] for call in progress.call_args_list]
    assert stages == ["Query processing", "Retrieval", "Reranking", "Context selection", "Answer generation", "Citation validation"]
    assert answer["metrics"]["trace_id"]
    assert answer["debug"]["configuration"] == asdict(settings)
    assert answer["debug"]["context_stats"]["characters"] <= settings.context_chars
    assert all(v >= 0 for v in answer["debug"]["timings"].values())


def test_jobs_persist_success_duplicate_and_failure(store, settings, service):
    jobs = IngestionJobs(store, service_factory=lambda _: service)
    try:
        first = jobs.submit("queued.txt", b"MapReduce is scalable.", settings)
        jobs.futures[first].result(timeout=10)
        assert store.jobs()[0]["status"] == "completed" and len(store.documents()) == 1
        duplicate = jobs.submit("same.txt", b"MapReduce is scalable.", settings)
        jobs.futures[duplicate].result(timeout=10)
        assert store.jobs()[0]["payload"]["duplicate"] and len(store.documents()) == 1
        failed = jobs.submit("empty.txt", b"    ", settings)
        jobs.futures[failed].result(timeout=10)
        assert store.jobs()[0]["status"] == "failed"
        assert len(Store(store.root).jobs()) == 3
        with pytest.raises(RAGError):
            jobs.submit("unsafe.exe", b"code", settings)
    finally:
        jobs.close()


def test_jobs_mark_restart_interruption(store):
    identifier = store.new_job("old.txt", {})
    store.update_job(identifier, "running", 0.2)
    jobs = IngestionJobs(store)
    try:
        assert store.jobs()[0]["status"] == "interrupted"
        assert "Submit" in store.jobs()[0]["payload"]["message"]
    finally:
        jobs.close()


def test_job_queue_is_bounded(store, settings):
    import threading
    gate = threading.Event()
    def slow(_):
        gate.wait(timeout=10)
        return None
    jobs = IngestionJobs(store, service_factory=slow)
    try:
        identifiers = [jobs.submit(f"{i}.txt", f"document {i}".encode(), settings) for i in range(10)]
        with pytest.raises(RAGError, match="full"):
            jobs.submit("overflow.txt", b"too many", settings)
        assert jobs.cancel(identifiers[-1])
        assert not jobs.cancel("not a job")
    finally:
        gate.set()
        jobs.close()


def test_settings_and_scope_survive_restart(store, settings):
    store.save_preference("settings", asdict(settings))
    store.save_preference("conversation:example", {"documents": ["specific"]})
    restored = Store(store.root)
    assert Settings(**restored.preference("settings")) == settings
    assert restored.preference("conversation:example")["documents"] == ["specific"]


def test_feedback_and_saved_answers_cascade_with_chat(corpus, settings, service):
    store, _ = corpus
    session = store.new_session()
    answer = Pipeline(store, settings, service).ask("MapReduce")
    store.add_turn(session, "MapReduce", answer)
    message_id = store.history(session)[-1]["message_id"]
    store.set_feedback(message_id, -1, "Needs more detail", True)
    assert Store(store.root).feedback()[message_id]["note"] == "Needs more detail"
    store.clear_session(session)
    assert not store.feedback()
    with pytest.raises(sqlite3.IntegrityError):
        store.set_feedback(message_id, 1)


def test_evaluation_history_has_reproducibility(corpus, settings, service):
    store, docs = corpus
    result = compare(store, settings, service, [{"question": "MapReduce", "relevant_source_ids": [docs[0]["id"]]}], k=1)
    assert result["run_id"] and result["configuration"] == asdict(settings)
    persisted = Store(store.root).evaluation_runs()
    assert persisted[0]["id"] == result["run_id"]
    assert persisted[0]["result"]["dataset_fingerprint"] == result["dataset_fingerprint"]
    assert persisted[0]["result"]["library_revision"] == store.revision()


def test_heading_cannot_hide_uncited_claim():
    valid, _ = validate_citations("A supported fact [S1].\n\n## A heading\nAn unsupported factual statement should not evade validation.",
                                 [{"citation_id": "S1", "id": "a"}])
    assert not valid


def test_highlighting_escapes_untrusted_html():
    html = highlighted_excerpt('<script>alert("attack")</script> MapReduce', "MapReduce")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<mark>MapReduce</mark>" in html


def test_chunks_preserve_reading_order(store, settings, service):
    content = "# First\nFirst section text.\n# Second\nSecond section text.\n# Third\nThird section text."
    Library(store, settings, service).ingest("ordered.md", content.encode())
    assert [c["section"] for c in store.chunks()] == ["First", "Second", "Third"]


def test_all_workspace_pages_render_and_search(tmp_path):
    from unittest.mock import patch
    with patch.dict("os.environ", {"RAG_DATA_DIR": str(tmp_path), "GEMINI_API_KEY": ""}):
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
        next(b for b in app.button if b.label == "Load example documents").click().run()
        for name in ["Documents", "Search studio", "Evaluation", "Activity", "Settings", "Workspace"]:
            app.radio(key="navigation").set_value(name).run()
            assert not app.exception, name
        app.radio(key="navigation").set_value("Search studio").run()
        next(t for t in app.text_input if t.label == "Search query").set_value("MapReduce")
        next(b for b in app.button if b.label == "Run search experiment").click().run()
        assert not app.exception and len(app.dataframe) == 2
        app.radio(key="navigation").set_value("Settings").run()
        next(n for n in app.number_input if n.label == "Evidence passages").set_value(3)
        next(b for b in app.button if b.label == "Save configuration").click().run()
        assert not app.exception and Store(tmp_path).preference("settings")["top_k"] == 3


def test_conversations_keep_independent_source_selection(tmp_path):
    from unittest.mock import patch
    with patch.dict("os.environ", {"RAG_DATA_DIR": str(tmp_path), "GEMINI_API_KEY": ""}):
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
        next(b for b in app.button if b.label == "Load example documents").click().run()
        first = app.selectbox(key="session_id").value
        first_source = app.multiselect(key="selected_docs").options[0]
        # AppTest multiselect values use the formatted filename.
        app.multiselect(key="selected_docs").select(first_source).run()
        app.multiselect(key="selected_docs").set_value([]).run()
        next(b for b in app.button if b.label == "New conversation").click().run()
        assert len(app.multiselect(key="selected_docs").value) == 3
        app.selectbox(key="session_id").set_value(first).run()
        assert not app.multiselect(key="selected_docs").value and app.chat_input[0].disabled
