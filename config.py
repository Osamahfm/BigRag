"""Validated settings shared by CLI, UI, and pipeline."""
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_strategy: str = "recursive"
    parent_size: int = 2400
    embedding_model: str = "gemini-embedding-001"
    dimension: int = 768
    generation_model: str = "gemini-3.5-flash-lite"
    backend: str = "gemini"
    top_k: int = 5
    candidate_k: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    reranker: str = "lexical"
    context_chars: int = 14000
    max_upload_mb: int = 25
    ocr: bool = False
    agent_steps: int = 3
    dense_weight: float = 0.6
    diversity: bool = True
    mmr_lambda: float = 0.75
    max_chunks_per_document: int = 0

    def __post_init__(self):
        if not self.embedding_model.strip() or not self.generation_model.strip():
            raise ValueError("Embedding and generation models must be nonempty.")
        if not 100 <= self.chunk_size <= 8000 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk size must be 100–8000; overlap must be smaller.")
        if self.parent_size < self.chunk_size:
            raise ValueError("Parent size must be at least the child size.")
        if self.chunk_strategy not in {"fixed", "recursive", "sentence", "semantic", "parent-child"}:
            raise ValueError("Unknown chunk strategy.")
        if not 1 <= self.dimension <= 3072 or self.backend not in {"demo", "gemini"}:
            raise ValueError("Invalid embedding settings.")
        if not 1 <= self.top_k <= self.candidate_k <= 100:
            raise ValueError("Retrieval limits must satisfy 1 <= top_k <= candidates <= 100.")
        if not 0 <= self.bm25_b <= 1 or self.bm25_k1 <= 0 or self.rrf_k < 1:
            raise ValueError("Invalid BM25 / RRF parameters.")
        if self.reranker not in {"none", "lexical", "cross-encoder"}:
            raise ValueError("Unknown reranker.")
        if not 1 <= self.agent_steps <= 5 or self.context_chars < 100 or self.max_upload_mb < 1:
            raise ValueError("Invalid resource limits.")
        if not 0 <= self.dense_weight <= 1 or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("Fusion weight and diversity balance must be between 0 and 1.")
        if not 0 <= self.max_chunks_per_document <= 100:
            raise ValueError("Document evidence cap must be between 0 and 100; 0 means unlimited.")

    def index_config(self):
        keys = ("chunk_size", "chunk_overlap", "chunk_strategy", "parent_size", "ocr")
        return {"schema": 2, **{k: asdict(self)[k] for k in keys}}

    def embedding_config(self):
        return {"schema": 1, "backend": self.backend, "model": self.embedding_model,
                "dimension": self.dimension, "task": "RETRIEVAL_DOCUMENT"}


def data_directory():
    folder = Path(os.getenv("RAG_DATA_DIR", "data"))
    return (ROOT / folder).resolve() if not folder.is_absolute() else folder.resolve()
