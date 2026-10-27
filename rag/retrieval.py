"""Normalized FAISS cosine + BM25Plus + weighted reciprocal rank fusion."""
import hashlib
import json
import os
import uuid

import faiss
import numpy as np
from rank_bm25 import BM25Plus

from rag.errors import RAGError
from rag.text import tokens


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(values, count, dimension):
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.shape != (count, dimension) or not np.isfinite(values).all():
        raise RAGError("Embedding dimensions or values do not match settings; rebuild the index.")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise RAGError("Embedding contains a zero vector.")
    faiss.normalize_L2(values)
    return values


def rrf(rankings, k=60, weights=None):
    weights = [1.0]*len(rankings) if weights is None else weights
    if len(weights) != len(rankings) or any(not np.isfinite(w) or w < 0 for w in weights) or (rankings and sum(weights) <= 0):
        raise ValueError("RRF weights must match rankings, be finite/nonnegative and have a positive total.")
    scores, items = {}, {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if not weight:
            continue
        seen = set()
        for rank, item in enumerate(ranking, 1):
            identifier = item["id"]
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0) + weight / (k + rank)
            items.setdefault(identifier, dict(item))
            for name in ("dense_score", "sparse_score"):
                if name in item:
                    items[identifier][name] = item[name]
    return [{**items[i], "score": scores[i], "score_type": "rrf"}
            for i in sorted(scores, key=lambda i: (-scores[i], i))]


class Retriever:
    def __init__(self, store, settings, service):
        self.store, self.settings, self.service = store, settings, service
        self.folder = store.root / "indexes"
        self.folder.mkdir(exist_ok=True)
        self.warnings = []

    def vectors(self, chunks):
        config = key(self.settings.embedding_config())
        cached = {}
        with self.store.connect() as db:
            for row in db.execute("SELECT chunk_id,dimension,vector FROM embeddings WHERE config=?", (config,)):
                if row["dimension"] == self.settings.dimension:
                    values = np.frombuffer(row["vector"], dtype=np.float32)
                    if values.shape == (self.settings.dimension,) and np.isfinite(values).all() and np.linalg.norm(values) > 0:
                        cached[row["chunk_id"]] = values
        missing = [c for c in chunks if c["id"] not in cached]
        if missing:
            arrays = normalize(self.service.embed([c["text"] for c in missing], "RETRIEVAL_DOCUMENT"),
                               len(missing), self.settings.dimension)
            with self.store.connect() as db:
                for chunk, array in zip(missing, arrays, strict=True):
                    # Do not resurrect embeddings for a document deleted during an API call.
                    if db.execute("SELECT 1 FROM chunks WHERE id=?", (chunk["id"],)).fetchone():
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?)",
                                   (chunk["id"], config, self.settings.dimension, array.tobytes()))
                    cached[chunk["id"]] = array
        return normalize([cached[c["id"]] for c in chunks], len(chunks), self.settings.dimension)

    def index(self, chunks):
        manifest_key = key({"embedding": self.settings.embedding_config(), "ids": [c["id"] for c in chunks]})
        path, metadata_path = self.folder / f"{manifest_key}.faiss", self.folder / f"{manifest_key}.json"
        try:
            manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw = path.read_bytes()
            if manifest != {"key": manifest_key, "sha256": hashlib.sha256(raw).hexdigest()}:
                raise ValueError("Checksum mismatch")
            index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
            if not isinstance(index, faiss.IndexFlatIP) or index.d != self.settings.dimension or index.ntotal != len(chunks):
                raise ValueError("Incompatible index")
            values = index.reconstruct_n(0, index.ntotal)
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
                raise ValueError("Invalid normalized vectors")
            return index
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass
        revision = self.store.revision()
        index = faiss.IndexFlatIP(self.settings.dimension)
        index.add(self.vectors(chunks))
        if revision != self.store.revision():
            raise RAGError("Library changed during indexing. Retry your search.")
        raw = faiss.serialize_index(index).tobytes()
        metadata = json.dumps({"key": manifest_key, "sha256": hashlib.sha256(raw).hexdigest()}).encode()
        for destination, content in ((path, raw), (metadata_path, metadata)):
            temp = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temp.write_bytes(content)
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
        return index

    def query_vector(self, query):
        cache_key = key({"config": self.settings.embedding_config(), "task": "RETRIEVAL_QUERY", "query": query})
        with self.store.connect() as db:
            row = db.execute("SELECT dimension,vector FROM query_cache WHERE key=?", (cache_key,)).fetchone()
        if row and row["dimension"] == self.settings.dimension:
            try:
                return normalize(np.frombuffer(row["vector"], dtype=np.float32).reshape(1, -1), 1, self.settings.dimension)
            except (ValueError, RAGError):
                pass
        vector = normalize(self.service.embed([query], "RETRIEVAL_QUERY"), 1, self.settings.dimension)
        with self.store.connect() as db:
            db.execute("INSERT OR REPLACE INTO query_cache VALUES(?,?,?)", (cache_key, self.settings.dimension, vector.tobytes()))
            # Bound the cache instead of retaining arbitrary user queries forever.
            db.execute("DELETE FROM query_cache WHERE rowid NOT IN (SELECT rowid FROM query_cache ORDER BY rowid DESC LIMIT 500)")
        return vector

    def dense(self, query, chunks, limit):
        index = self.index(chunks)
        scores, positions = index.search(self.query_vector(query), min(limit, len(chunks)))
        return [{**chunks[int(i)], "score": float(score), "dense_score": float(score), "score_type": "cosine"}
                for i, score in zip(positions[0], scores[0], strict=True) if i >= 0 and score > 0]

    def sparse(self, query, chunks, limit):
        query_tokens = tokens(query)
        # BM25Plus positive IDF behaves predictably on tiny lecture collections.
        with self.store.connect() as db:
            persisted = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,tokens FROM chunks")}
        corpus = [persisted.get(c["id"], tokens(c["text"])) or ["__empty__"] for c in chunks]
        model = BM25Plus(corpus, k1=self.settings.bm25_k1, b=self.settings.bm25_b, delta=1)
        scores = model.get_scores(query_tokens)
        overlap = set(query_tokens)
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["id"]))
        return [{**chunks[i], "score": float(scores[i]), "sparse_score": float(scores[i]), "score_type": "bm25plus"}
                for i in ranked if overlap.intersection(corpus[i])][:limit]

    def search(self, query, mode="hybrid", document_ids=None, limit=None):
        if mode not in {"dense", "sparse", "hybrid"}:
            raise RAGError("Unknown retrieval method.")
        if not query.strip() or len(query) > 4000:
            raise RAGError("Search query must contain 1–4000 characters.")
        chunks = self.store.chunks(document_ids)
        if not chunks:
            return []
        limit = min(limit or self.settings.candidate_k, 100)
        if mode == "sparse":
            return self.sparse(query, chunks, limit)
        if mode == "dense":
            return self.dense(query, chunks, limit)
        sparse = self.sparse(query, chunks, limit)
        if self.settings.dense_weight == 0:
            return rrf([sparse], self.settings.rrf_k)[:limit]
        try:
            dense = self.dense(query, chunks, limit)
        except RAGError:
            self.warnings.append("Dense retrieval unavailable; hybrid search used keyword results.")
            return sparse
        return rrf([dense, sparse], self.settings.rrf_k,
                   [self.settings.dense_weight, 1-self.settings.dense_weight])[:limit]
