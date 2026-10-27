import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from conftest import fake_generation

from rag.agent import RetrievalAgent
from rag.embeddings import GeminiService, api_call
from rag.errors import RAGError
from rag.evaluation import compare, generation_proxies, retrieval_metrics
from rag.generation import UNSUPPORTED, build_context, validate_citations
from rag.pipeline import MODES, Pipeline
from rag.query import expand, rewrite, route
from rag.retrieval import Retriever
from rag.storage import Store


@pytest.mark.parametrize("mode", MODES)
def test_real_end_to_end_each_mode(corpus, settings, service, mode):
    store, docs = corpus
    answer = Pipeline(store, settings, service).ask("What are the advantages of MapReduce?", mode=mode,
                                                 document_ids=[docs[0]["id"]])
    assert answer["cited_sources"] and "Source: lecture.pdf, Page " in answer["content"]
    assert all(c["document_id"] == docs[0]["id"] for c in answer["sources"])
    assert answer["metrics"]["total_ms"] >= 0
    assert store.events()[0]["kind"] == "query"
    if mode == "Multi-Query":
        assert len(answer["debug"]["expanded_queries"]) > 1
    if mode == "Agentic":
        assert answer["debug"]["agent_decisions"]
    if mode == "Self-Reflective":
        assert answer["verification"]["status"] == "model judged"


def test_parent_context_is_larger(settings, service, store):
    from rag.library import Library
    Library(store, settings, service).ingest("parent.txt", ("MapReduce is scalable. "*30).encode())
    pipeline = Pipeline(store, settings, service)
    _, child, _, _ = pipeline.retrieve("MapReduce scalability", mode="Conversational")
    _, parent, _, _ = pipeline.retrieve("MapReduce scalability", mode="Parent-Document")
    assert len(parent[0]["text"]) > len(child[0]["text"])
    assert len({c["parent_id"] for c in parent}) == len(parent)


def test_rewrite_only_when_needed(service):
    history = [{"role": "user", "content": "What is MapReduce?"}]
    service.generate.side_effect = fake_generation
    assert rewrite("How does photosynthesis work?", history, service) == "How does photosynthesis work?"
    assert rewrite("What are the advantages of Spark?", history, service) == "What are the advantages of Spark?"
    service.generate.assert_not_called()
    assert "MapReduce" in rewrite("What are its advantages?", history, service)
    calls = service.generate.call_count
    assert rewrite("What are its advantages?", history, service, enabled=False) == "What are its advantages?"
    assert service.generate.call_count == calls
    service.generate.side_effect = RAGError("offline")
    assert rewrite("How does it work?", history, service) == "How does it work?"


def test_expand_and_route(service):
    assert len(expand("MapReduce advantages", service)) == 2
    assert expand("query", service, enabled=False) == ["query"]
    assert route('Find "TASK-42"') == "sparse"
    assert route("Explain this concept") == "dense"
    assert route("Compare storage options") == "hybrid"
    service.generate.return_value = "not JSON"
    service.generate.side_effect = None
    assert expand("query", service, multi=True) == ["query"]


@pytest.mark.parametrize("answer", ["", "uncited fact", "fact [S999]", "fact [S1] [Source: invented.pdf, Page 9]",
    "fact [S1] [Sx]", "fact [S01]", "fact [S1] [S2]", "fact [S1", "fact [S1]"])
def test_citation_identity_validation(answer):
    source = {"id": "a", "citation_id": "S1", "document_id": "d", "source": "file.pdf", "page": 1,
              "section": "Page 1", "text": "fact"}
    valid, cited = validate_citations(answer, [source])
    assert valid == (answer == "fact [S1]")
    assert bool(cited) == valid


def test_grounding_fail_closed_and_prompt_isolation(corpus, settings, service):
    store, _ = corpus
    service.generate.side_effect = None
    service.generate.return_value = "Fabricated [S999]"
    answer = Pipeline(store, settings, service).ask("MapReduce")
    assert answer["content"] == UNSUPPORTED and not answer["cited_sources"]
    prompt, instruction = service.generate.call_args.args
    assert "untrusted" in instruction and "history" not in json.loads(prompt)


def test_reflection_revises_once_then_abstains(corpus, settings, service):
    store, _ = corpus
    service.generate.side_effect = ["Claim [S1]", '{"supported": false, "reason": "not supported"}',
                                    "Revised [S1]", '{"supported": false, "reason": "still unsupported"}']
    answer = Pipeline(store, settings, service).ask("MapReduce", mode="Self-Reflective", expansion=False)
    assert answer["content"] == UNSUPPORTED and service.generate.call_count == 4


def test_reflection_unavailable_abstains(corpus, settings, service):
    store, _ = corpus
    service.generate.side_effect = ["Claim [S1]", "bad json"]
    answer = Pipeline(store, settings, service).ask("MapReduce", mode="Self-Reflective", expansion=False)
    assert answer["verification"]["status"] == "reflection unavailable; abstained"


def test_corrective_retries_and_abstains(corpus, settings, service):
    store, _ = corpus
    pipeline = Pipeline(store, settings, service)
    answer = pipeline.ask("Unrelated galactic teleportation", mode="Corrective", retrieval="sparse")
    assert answer["debug"]["corrective_retry"] and answer["content"] == UNSUPPORTED
    assert not answer["sources"]


def test_agent_limits_validation_and_real_metadata(corpus, settings, service):
    store, docs = corpus
    agent = RetrievalAgent(Retriever(store, settings, service), max_steps=1)
    found, trace = agent.run("Compare MapReduce and Spark")
    assert found and len(trace) == 1 and trace[0]["limitation"]
    with pytest.raises(RAGError, match="iteration"):
        agent.call("hybrid", "test")
    agent = RetrievalAgent(Retriever(store, settings, service))
    with pytest.raises(RAGError, match="arguments"):
        agent.call("execute", "malicious")
    with pytest.raises(RAGError, match="unknown document"):
        agent.call("hybrid", "test", ["missing"])
    found, trace = agent.run("How many pages?", [docs[0]["id"]])
    assert trace[0]["tool"] == "metadata" and json.loads(found[0]["text"])["pages"] == 3
    assert agent.run("MapReduce", [docs[0]["id"]])[1][0]["tool"] == "hybrid"


def test_context_budget(corpus, settings, service):
    store, _ = corpus
    candidates = Retriever(store, settings, service).search("MapReduce")
    assert sum(len(s["text"]) for s in build_context(candidates, replace(settings, context_chars=100))) <= 100


def test_persistent_sessions_and_source_cleanup(corpus, settings, service):
    store, docs = corpus
    first, second = store.new_session(), store.new_session("Other")
    answer = Pipeline(store, settings, service).ask("MapReduce", document_ids=[docs[0]["id"]])
    store.add_turn(first, "MapReduce", answer)
    reloaded = Store(store.root)
    assert len(reloaded.history(first)) == 2 and not reloaded.history(second)
    assert any(s["title"] == "MapReduce" for s in reloaded.sessions())
    store.rename_session(first, "Lecture notes")
    store.delete_document(docs[0]["id"])
    retained = store.history(first)[-1]
    assert not retained["sources"] and not retained["cited_sources"] and "debug" not in retained
    assert retained["source_notice"]
    store.clear_session(first)
    assert not store.history(first)


@pytest.mark.parametrize("code,calls", [(429, 3), (503, 3), (401, 1), (403, 1), (404, 1)])
def test_api_retry_and_secret_redaction(code, calls):
    error = Exception("SECRET_KEY")
    error.code = code
    operation, sleep = Mock(side_effect=error), Mock()
    with pytest.raises(RAGError) as exc:
        api_call(operation, sleep)
    assert operation.call_count == calls and sleep.call_count == calls-1
    assert "SECRET" not in str(exc.value)


def test_retry_transport_then_success():
    operation = Mock(side_effect=[httpx.ConnectError("network"), "ok"])
    assert api_call(operation, Mock()) == "ok"


def test_gemini_embedding_sdk_batch_types_and_dimension():
    client = Mock()
    client.models.embed_content.side_effect = lambda **kw: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1., 0., 0.]) for _ in kw["contents"]])
    service = GeminiService(replace(__import__("config").Settings(), dimension=3), client=client)
    result = service.embed(["text"]*33, "RETRIEVAL_DOCUMENT")
    assert result.shape == (33, 3) and result.dtype == np.float32
    assert client.models.embed_content.call_count == 2
    assert client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
    client.models.embed_content.side_effect = None
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[])
    with pytest.raises(RAGError, match="invalid embeddings"):
        service.embed(["text"], "RETRIEVAL_QUERY")


def test_no_key_is_actionable():
    with pytest.raises(RAGError, match="missing"):
        GeminiService(__import__("config").Settings(), api_key="")


def test_pipeline_logs_failure(corpus, settings, service):
    store, _ = corpus
    service.generate.side_effect = RAGError("Gemini temporarily unavailable")
    with pytest.raises(RAGError):
        Pipeline(store, settings, service).ask("MapReduce")
    assert store.events()[0]["kind"] == "error"


def test_metric_formulas():
    result = retrieval_metrics(["wrong", "a", "a", "b"], {"a", "b"}, 3)
    assert result["precision@k"] == pytest.approx(2/3)
    assert result["recall@k"] == 1 and result["mrr"] == 0.5 and result["hit_rate@k"] == 1
    assert 0 < result["ndcg@k"] < 1
    assert retrieval_metrics([], {"a"}, 3)["hit_rate@k"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics([], set(), 3)


def test_evaluation_real_retrieval(corpus, settings, service):
    store, docs = corpus
    dataset = [{"question": "MapReduce scalability", "reference_answer": "MapReduce scales.",
                "relevant_source_ids": [docs[0]["id"]]}]
    measured = compare(store, settings, service, dataset, k=1, generation=True)
    assert len(measured["summary"]) == 4 and len(measured["details"]) == 4
    assert all(r["recall@k"] == 1 for r in measured["summary"])
    assert "reference_token_f1_proxy" in measured["summary"][0]
    assert not measured["model_judge"]
    with pytest.raises(RAGError, match="missing"):
        compare(store, settings, service, [{"question": "a", "relevant_sources": ["missing.txt"]}])


def test_evaluation_never_labels_fallback_as_hybrid(corpus, settings, service):
    store, docs = corpus
    service.embed.side_effect = RAGError("offline")
    with pytest.raises(RAGError):
        compare(store, settings, service, [{"question": "MapReduce", "relevant_source_ids": [docs[0]["id"]]}])


def test_generation_metrics_are_proxies():
    values = generation_proxies("same words", "same words", [{"id": "a", "text": "same words"}], [{"id": "a"}])
    assert values["reference_token_f1_proxy"] == 1 and values["citation_validity"] == 1
