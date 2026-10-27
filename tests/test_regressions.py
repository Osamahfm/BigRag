from dataclasses import replace
from unittest.mock import Mock

import pytest

from config import Settings
from rag.embeddings import GeminiService
from rag.errors import RAGError
from rag.generation import validate_citations
from rag.library import Library
from rag.pipeline import Pipeline


def test_uncited_paragraph_rejected():
    source = {"id": "a", "citation_id": "S1"}
    valid, _ = validate_citations("A supported claim with evidence [S1].\n\nThis second paragraph invents a new unsupported factual assertion.", [source])
    assert not valid


def test_deletion_removes_nonselected_candidate_trace(corpus, settings, service):
    store, docs = corpus
    session = store.new_session()
    answer = Pipeline(store, replace(settings, top_k=1), service).ask("MapReduce photosynthesis")
    final_doc = answer["sources"][0]["document_id"]
    victim = next(d["id"] for d in docs if d["id"] != final_doc)
    # A candidate that did not make the final context still contains document text.
    answer["debug"]["candidates"].append(store.chunks([victim])[0])
    store.add_turn(session, "MapReduce photosynthesis", answer)
    store.delete_document(victim)
    assert "debug" not in store.history(session)[-1]
    assert store.history(session)[-1]["sources"]


def test_failed_reindex_preserves_existing_document(corpus, settings, service):
    store, _ = corpus
    doc = Library(store, settings, service).ingest("reindex.txt", b"First sentence about MapReduce. Second sentence about scalability.")
    original = store.chunks([doc["id"]])
    service.embed.side_effect = RAGError("unavailable")
    with pytest.raises(RAGError):
        Library(store, replace(settings, chunk_strategy="semantic"), service).reindex(doc["id"])
    assert store.chunks([doc["id"]]) == original


def test_api_error_observer_is_redacted(store):
    client = Mock()
    error = Exception("API_KEY_SECRET")
    error.code = 401
    client.models.generate_content.side_effect = error
    service = GeminiService(Settings(), client=client)
    service.error_observer = lambda event: store.event("api_error", event)
    with pytest.raises(RAGError):
        service.generate("question", "instruction")
    assert service.api_errors == 1
    assert store.events()[0]["kind"] == "api_error"
    assert store.events()[0]["code"] == 401 and "SECRET" not in str(store.events())
