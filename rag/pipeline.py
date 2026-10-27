import re
import time
import uuid
from dataclasses import asdict

from rag.agent import RetrievalAgent
from rag.diversity import diversify
from rag.errors import RAGError
from rag.generation import build_context, generate
from rag.query import expand, rewrite, route
from rag.reranking import rerank
from rag.retrieval import Retriever, rrf
from rag.text import tokens

MODES = ["Naive", "Conversational", "Hybrid", "Multi-Query", "Parent-Document", "Corrective", "Self-Reflective", "Agentic"]


class Pipeline:
    def __init__(self, store, settings, service):
        self.store, self.settings, self.service = store, settings, service
        self.retriever = Retriever(store, settings, service)
        if hasattr(service, "error_observer"):
            service.error_observer = lambda event: store.event("api_error", event)

    def retrieve(self, question, history=None, mode="Hybrid", retrieval="hybrid", document_ids=None,
                 rewriting=True, expansion=True, routing=False, progress=None):
        if mode not in MODES:
            raise RAGError("Unknown RAG mode.")
        question = question.strip()
        if not question or len(question) > 4000:
            raise RAGError("Question must contain 1–4000 characters.")
        if document_ids is not None:
            known = {d["id"] for d in self.store.documents()}
            if not set(document_ids) <= known:
                raise RAGError("Selected documents were removed. Refresh the library.")
        started = time.perf_counter()
        notify = progress or (lambda stage, detail: None)
        notify("Query processing", "Resolving references and preparing search variants")
        query = rewrite(question, history or [], self.service, rewriting and mode != "Naive")
        method = "dense" if mode == "Naive" else ("hybrid" if mode == "Hybrid" else retrieval)
        if routing and mode not in {"Naive", "Hybrid"}:
            method = route(query)
        queries = expand(query, self.service, multi=mode == "Multi-Query", enabled=expansion and mode != "Naive")
        query_ms = (time.perf_counter()-started)*1000
        search_started = time.perf_counter()
        notify("Retrieval", f"Searching with {method} retrieval")
        self.retriever.warnings = []
        decisions = []
        if mode == "Agentic":
            candidates, decisions = RetrievalAgent(self.retriever, self.settings.agent_steps).run(query, document_ids)
        elif re.search(r"summari[sz]e|summary|لخص|ملخص", query, re.I):
            # Representative sections cover the selected corpus even without lexical query matches.
            all_chunks = self.store.chunks(document_ids)
            groups = {}
            for c in all_chunks:
                groups.setdefault(c["document_id"], []).append(c)
            candidates = []
            for i in range(self.settings.candidate_k):
                for chunks in groups.values():
                    if i < len(chunks):
                        candidates.append({**chunks[i], "score": 1 / (i+1), "score_type": "summary coverage"})
            candidates = candidates[:self.settings.candidate_k]
            decisions.append({"tool": "representative sections", "limitation": "Summaries cover bounded selected excerpts, not every page."})
        else:
            rankings = [self.retriever.search(q, method, document_ids) for q in queries]
            candidates = rankings[0] if len(rankings) == 1 else rrf(rankings, self.settings.rrf_k)
        corrective = False
        coverage = max((len(set(tokens(query)).intersection(tokens(c["text"]))) / max(1, len(set(tokens(query))))
                        for c in candidates), default=0)
        if mode == "Corrective" and coverage < 0.25:
            revised = " ".join(tokens(query))
            if revised:
                retry = self.retriever.search(revised, "hybrid", document_ids)
                candidates = rrf([candidates, retry], self.settings.rrf_k)
                queries.append(revised)
                corrective = True
            if not candidates or max((len(set(tokens(query)).intersection(tokens(c["text"])))
                                       for c in candidates), default=0) == 0:
                candidates = []
        candidates = candidates[:self.settings.candidate_k]
        search_ms = (time.perf_counter()-search_started)*1000
        rerank_started = time.perf_counter()
        notify("Reranking", f"Reviewing {len(candidates)} candidates")
        reranked, warning = rerank(query, candidates, "none" if mode == "Naive" else self.settings.reranker)
        if warning:
            self.retriever.warnings.append(warning)
        rerank_ms = (time.perf_counter()-rerank_started)*1000
        selection_started = time.perf_counter()
        notify("Context selection", "Balancing relevant evidence and repeated passages")
        diversified, diversity_method = (reranked, "disabled") if mode == "Naive" else diversify(reranked, self.store, self.settings)
        context = build_context(diversified, self.settings, parent=mode == "Parent-Document")
        def trace_chunks(chunks):
            return [{**{k: v for k, v in c.items() if k not in {"parent_text", "text"}},
                     "text": c["text"][:1000], "excerpt_truncated": len(c["text"]) > 1000} for c in chunks]
        debug = {"original_query": question, "rewritten_query": query, "expanded_queries": queries,
                 "retrieval_method": method, "mode": mode, "candidates": trace_chunks(candidates),
                 "reranked": trace_chunks(reranked), "diversified": trace_chunks(diversified), "context": context, "corrective_retry": corrective,
                 "agent_decisions": decisions, "warnings": self.retriever.warnings,
                 "diversity_method": diversity_method, "configuration": asdict(self.settings),
                 "timings": {"query_processing_ms": query_ms, "search_ms": search_ms,
                             "rerank_ms": rerank_ms, "context_ms": (time.perf_counter()-selection_started)*1000},
                 "context_stats": {"characters": sum(len(c["text"]) for c in context),
                                   "documents": len({c["document_id"] for c in context}),
                                   "budget": self.settings.context_chars}}
        return query, context, debug, time.perf_counter() - started

    def ask(self, question, history=None, mode="Hybrid", retrieval="hybrid", document_ids=None,
            detail="concise", progress=None, **options):
        started = time.perf_counter()
        tokens_before = getattr(self.service, "usage_tokens", 0)
        try:
            query, sources, debug, retrieval_time = self.retrieve(question, history, mode, retrieval, document_ids, progress=progress, **options)
            generation_start = time.perf_counter()
            if progress:
                progress("Answer generation", f"Writing from {len(sources)} selected evidence passages")
            content, cited, verification = generate(question, query, sources, self.service, detail, mode == "Self-Reflective")
            if progress:
                progress("Citation validation", verification["status"])
            metrics = {"retrieval_ms": round(retrieval_time*1000, 2),
                       "generation_ms": round((time.perf_counter()-generation_start)*1000, 2),
                       "total_ms": round((time.perf_counter()-started)*1000, 2), "sources": len(sources),
                       "tokens": getattr(self.service, "usage_tokens", 0)-tokens_before,
                       "approx_context_tokens": sum(len(s["text"]) for s in sources)//4,
                       "mode": mode, "backend": self.settings.backend,
                       "trace_id": uuid.uuid4().hex, "abstained": not bool(cited),
                       "cited_documents": len({c["document_id"] for c in cited}),
                       "evidence_documents": len({c["document_id"] for c in sources})}
            self.store.event("query", metrics)
            return {"role": "assistant", "content": content, "query": query, "sources": sources,
                    "cited_sources": cited, "verification": verification, "metrics": metrics, "debug": debug}
        except RAGError as exc:
            self.store.event("error", {"message": str(exc), "mode": mode})
            raise
