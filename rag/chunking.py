"""Every child belongs to a same-page parent; metadata never moves across pages."""
import hashlib
import re

import numpy as np


def fixed(text, size, overlap):
    return [text[i:i+size].strip() for i in range(0, len(text), size-overlap) if text[i:i+size].strip()]


def sentence_split(text, size, overlap):
    sentences = re.split(r"(?<=[.!?؟。])\s+|\n\n+", text)
    result, current = [], ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                result.append(current)
                current = ""
            result.extend(fixed(sentence, size, overlap))
        elif current and len(current) + len(sentence) + 1 > size:
            result.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + sentence).strip()
            if len(current) > size:
                current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        result.append(current)
    return result


def split(text, settings, service=None):
    strategy = settings.chunk_strategy
    if strategy == "fixed":
        return fixed(text, settings.chunk_size, settings.chunk_overlap)
    if strategy == "sentence":
        return sentence_split(text, settings.chunk_size, settings.chunk_overlap)
    if strategy == "semantic":
        if service is None:
            raise ValueError("Semantic chunking needs an embedding service.")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?؟。])\s+|\n\n+", text) if s.strip()]
        if len(sentences) < 2:
            return fixed(text, settings.chunk_size, settings.chunk_overlap)
        vectors = np.asarray(service.embed(sentences, "SEMANTIC_SIMILARITY"), dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        groups, current = [], sentences[0]
        for i, sentence in enumerate(sentences[1:], 1):
            if float(vectors[i] @ vectors[i-1]) < 0.55 or len(current)+len(sentence)+1 > settings.chunk_size:
                groups.extend(fixed(current, settings.chunk_size, settings.chunk_overlap))
                current = sentence
            else:
                current += " " + sentence
        return groups + fixed(current, settings.chunk_size, settings.chunk_overlap)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap, separators=["\n\n", "\n", ". ", "؟ ", " ", ""]).split_text(text)


def chunk_units(units, document_id, settings, service=None):
    chunks = []
    for unit_index, unit in enumerate(units):
        # Parents always exist, allowing parent retrieval on any indexed corpus.
        for parent_index, parent in enumerate(fixed(unit["text"], settings.parent_size, 0)):
            parent_id = f"{document_id}:{unit_index}:{parent_index}"
            for child_index, text in enumerate(split(parent, settings, service)):
                chunk_id = hashlib.sha256(f"{parent_id}:{child_index}:{text}".encode()).hexdigest()
                chunks.append({**unit, "id": chunk_id, "document_id": document_id,
                    "text": text, "parent_id": parent_id, "parent_text": parent, "ordinal": len(chunks)})
    return chunks
