from functools import lru_cache

from rag.text import tokens


@lru_cache(maxsize=1)
def cross_encoder():
    from sentence_transformers import CrossEncoder
    # An installed local model is required. No surprise downloads during a question.
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", local_files_only=True)


def rerank(query, candidates, method="lexical"):
    if not candidates or method == "none":
        return candidates, None
    warning = None
    if method == "cross-encoder":
        try:
            scores = cross_encoder().predict([(query, c["text"]) for c in candidates])
            result = [{**c, "rerank_score": float(s), "reranker": "cross-encoder"} for c, s in zip(candidates, scores, strict=True)]
            return sorted(result, key=lambda c: -c["rerank_score"]), None
        except Exception:
            warning = "Local cross-encoder unavailable; using lexical coverage reranking. See optional setup."
    q = set(tokens(query))
    result = []
    for rank, candidate in enumerate(candidates, 1):
        coverage = len(q.intersection(tokens(candidate["text"]))) / max(1, len(q))
        result.append({**candidate, "rerank_score": coverage + 0.05 / rank, "reranker": "lexical coverage"})
    return sorted(result, key=lambda c: (-c["rerank_score"], c["id"])), warning
