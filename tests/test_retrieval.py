from dataclasses import replace

import numpy as np
import pytest
from conftest import make_pdf

from rag.errors import RAGError
from rag.library import Library
from rag.reranking import rerank
from rag.retrieval import Retriever, normalize, rrf
from rag.storage import Store


@pytest.mark.parametrize("method", ["dense", "sparse", "hybrid"])
def test_retrieval_and_filters(corpus, settings, service, method):
    store, docs = corpus
    retriever = Retriever(store, settings, service)
    found = retriever.search("MapReduce scalability", method)
    assert found[0]["source"] == "lecture.pdf"
    filtered = retriever.search("MapReduce", method, [docs[1]["id"]])
    assert all(c["document_id"] == docs[1]["id"] for c in filtered)
    assert retriever.search("MapReduce", method, []) == []


def test_normalized_persistent_index(corpus, settings, service):
    store, _ = corpus
    retriever = Retriever(store, settings, service)
    retriever.search("MapReduce", "dense")
    calls = service.embed.call_count
    reloaded = Retriever(Store(store.root), settings, service)
    reloaded.search("MapReduce", "dense")
    assert service.embed.call_count == calls
    index = reloaded.index(store.chunks())
    assert index.d == settings.dimension
    assert np.allclose(np.linalg.norm(index.reconstruct_n(0, index.ntotal), axis=1), 1)


def test_incremental_indexing_and_reindex_reuse(corpus, settings, service):
    store, docs = corpus
    retriever = Retriever(store, settings, service)
    retriever.search("MapReduce", "dense")
    service.embed.reset_mock()
    Library(store, settings, service).ingest("extra.txt", b"A new document about distributed systems.")
    retriever.search("MapReduce", "dense")
    document_calls = [call for call in service.embed.call_args_list if call.args[1] == "RETRIEVAL_DOCUMENT"]
    assert len(document_calls) == 1 and len(document_calls[0].args[0]) == 1
    service.embed.reset_mock()
    Library(store, settings, service).reindex(docs[0]["id"])
    retriever.search("MapReduce", "dense")
    assert not any(call.args[1] == "RETRIEVAL_DOCUMENT" for call in service.embed.call_args_list)


def test_embedding_config_invalidates(corpus, settings, service):
    store, _ = corpus
    Retriever(store, settings, service).search("MapReduce", "dense")
    service.embed.reset_mock()
    changed = replace(settings, embedding_model="new-model")
    Retriever(store, changed, service).search("MapReduce", "dense")
    assert any(c.args[1] == "RETRIEVAL_DOCUMENT" for c in service.embed.call_args_list)


@pytest.mark.parametrize("values", [[[0, 0]], [[float("nan"), 1]], [[1, 2, 3]]])
def test_invalid_embeddings(values):
    with pytest.raises(RAGError):
        normalize(values, 1, 2)


def test_corrupt_index_rebuilt_from_cached_vectors(corpus, settings, service):
    store, _ = corpus
    retriever = Retriever(store, settings, service)
    retriever.search("MapReduce", "dense")
    calls = service.embed.call_count
    for path in retriever.folder.glob("*.faiss"):
        path.write_bytes(b"broken")
    assert retriever.search("MapReduce", "dense")
    assert service.embed.call_count == calls


def test_hybrid_api_failure_falls_back(corpus, settings, service):
    store, _ = corpus
    service.embed.side_effect = RAGError("API unavailable")
    retriever = Retriever(store, settings, service)
    assert retriever.search("MapReduce", "hybrid")
    assert retriever.warnings
    with pytest.raises(RAGError):
        retriever.search("MapReduce", "dense")


def test_rrf_exact_values_and_deduplication():
    a, b = {"id": "a"}, {"id": "b"}
    result = rrf([[a, b], [b, a]], 60)
    assert len(result) == 2 and result[0]["score"] == pytest.approx(1/61+1/62)
    assert len(rrf([[a, a]], 60)) == 1


def test_rerank_local_and_optional_fallback():
    from unittest.mock import patch
    candidates = [{"id": "bad", "text": "biology", "score": 0.9},
                  {"id": "good", "text": "MapReduce scalability", "score": 0.1}]
    found, warning = rerank("MapReduce scalability", candidates)
    assert found[0]["id"] == "good" and warning is None
    assert rerank("MapReduce", candidates, "none")[0] == candidates
    with patch("rag.reranking.cross_encoder", side_effect=ImportError):
        found, warning = rerank("MapReduce scalability", candidates, "cross-encoder")
        assert found[0]["id"] == "good" and warning


def test_delete_purges_every_index_and_embedding(corpus, settings, service):
    store, docs = corpus
    Retriever(store, settings, service).search("MapReduce", "dense")
    doc_id = docs[0]["id"]
    store.delete_document(doc_id)
    assert all(c["document_id"] != doc_id for c in store.chunks())
    assert not list((store.root/"indexes").glob("*.faiss"))
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM embeddings e LEFT JOIN chunks c ON c.id=e.chunk_id WHERE c.id IS NULL").fetchone()[0] == 0
    assert not any(c["source"] == "lecture.pdf" for c in Retriever(store, settings, service).search("MapReduce", "hybrid"))
    with pytest.raises(RAGError):
        store.raw_document(doc_id)


def test_duplicate_documents(corpus, settings, service):
    store, docs = corpus
    result = Library(store, settings, service).ingest("other.pdf", make_pdf(["MapReduce processes data in parallel.", "", "MapReduce advantages include scalability and fault tolerance."]))
    assert result["duplicate"] and len(store.documents()) == len(docs)
