"""A bounded deterministic tool-selection agent. No arbitrary tools or execution."""
import json
import re

from rag.errors import RAGError
from rag.query import route
from rag.retrieval import rrf


class RetrievalAgent:
    def __init__(self, retriever, max_steps=3):
        if not 1 <= max_steps <= 5:
            raise ValueError("Agent step limit must be 1–5.")
        self.retriever, self.max_steps = retriever, max_steps
        self.decisions = []

    def call(self, tool, query, document_ids=None):
        if len(self.decisions) >= self.max_steps:
            raise RAGError("Agent iteration limit reached.")
        if tool not in {"dense", "sparse", "hybrid", "metadata"} or not isinstance(query, str) or len(query) > 4000:
            raise RAGError("Invalid agent tool arguments.")
        known = {d["id"] for d in self.retriever.store.documents()}
        if document_ids is not None and (not isinstance(document_ids, (list, set, tuple)) or not set(document_ids) <= known):
            raise RAGError("Agent selected an unknown document.")
        decision = {"tool": tool, "query": query, "document_ids": document_ids,
                    "reason": "Bounded local router chose an available tool."}
        self.decisions.append(decision)
        if tool == "metadata":
            docs = [d for d in self.retriever.store.documents() if document_ids is None or d["id"] in document_ids]
            results = [{"id": "metadata:"+d["id"], "document_id": d["id"], "source": d["source"],
                        "page": None, "section": "Library metadata", "parent_id": "metadata:"+d["id"],
                        "text": json.dumps({k: d.get(k) for k in ("source", "format", "bytes", "pages", "chunks", "created")}),
                        "parent_text": "", "score": 1.0, "score_type": "metadata"} for d in docs]
        else:
            results = self.retriever.search(query, tool, document_ids)
        decision["results"] = len(results)
        return results

    def run(self, query, document_ids=None):
        self.decisions = []
        if re.search(r"file size|filename|how many (files|documents|pages)|list (files|documents)|metadata|أسماء الملفات|عدد الصفحات", query, re.I):
            return self.call("metadata", query, document_ids), self.decisions
        if re.search(r"compare|versus|difference|قارن|مقارنة|الفرق", query, re.I):
            docs = [d for d in self.retriever.store.documents() if document_ids is None or d["id"] in document_ids]
            rankings = [self.call("hybrid", query, [doc["id"]]) for doc in docs[:self.max_steps]]
            if len(docs) > self.max_steps:
                self.decisions[-1]["limitation"] = "Comparison limited to selected documents within the step budget."
            return rrf(rankings, self.retriever.settings.rrf_k), self.decisions
        results = self.call(route(query), query, document_ids)
        if not results and len(self.decisions) < self.max_steps:
            results = self.call("hybrid", query, document_ids)
        return results, self.decisions
