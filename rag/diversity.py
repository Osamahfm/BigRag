"""Maximal marginal relevance: cached semantic vectors or local lexical TF-IDF.

The relevance term uses normalized original ranking; the redundancy term uses
pairwise cosine similarity. No additional Gemini calls are required.
"""
import math
from collections import Counter

import numpy as np

from rag.retrieval import key
from rag.text import tokens


def candidate_vectors(candidates, store, settings):
    cached = {}
    with store.connect() as db:
        for row in db.execute("SELECT chunk_id,dimension,vector FROM embeddings WHERE config=?",
                              (key(settings.embedding_config()),)):
            if row["dimension"] == settings.dimension:
                v = np.frombuffer(row["vector"], dtype=np.float32)
                if v.shape == (settings.dimension,) and np.isfinite(v).all() and np.linalg.norm(v) > 0:
                    cached[row["chunk_id"]] = v
    if candidates and all(c["id"] in cached for c in candidates):
        vectors = np.array([cached[c["id"]] for c in candidates], dtype=np.float32)
        method = "cached embedding cosine"
    else:
        counts = [Counter(tokens(c["text"])) for c in candidates]
        vocabulary = sorted({t for count in counts for t in count})
        df = Counter(t for count in counts for t in count)
        vectors = np.array([[count[t]*(math.log((1+len(counts))/(1+df[t]))+1) for t in vocabulary]
                            for count in counts], dtype=np.float32)
        method = "local TF-IDF cosine"
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8), method


def diversify(candidates, store, settings):
    if not candidates or not settings.diversity:
        return candidates, "disabled"
    vectors, method = candidate_vectors(candidates, store, settings)
    similarities = vectors @ vectors.T
    relevance = np.array([1/(i+1) for i in range(len(candidates))], dtype=np.float32)
    chosen, remaining, doc_counts, output = [], set(range(len(candidates))), Counter(), []
    while remaining:
        scored = []
        for i in remaining:
            doc = candidates[i]["document_id"]
            if settings.max_chunks_per_document and doc_counts[doc] >= settings.max_chunks_per_document:
                continue
            redundancy = max((float(similarities[i, j]) for j in chosen), default=0)
            score = settings.mmr_lambda*float(relevance[i]) - (1-settings.mmr_lambda)*max(0, redundancy)
            scored.append((score, -i, i))
        if not scored:
            break
        score, _, winner = max(scored)
        chosen.append(winner)
        remaining.remove(winner)
        doc_counts[candidates[winner]["document_id"]] += 1
        output.append({**candidates[winner], "mmr_score": score, "diversity_method": method})
    return output, method
