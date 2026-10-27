from rag.chunking import chunk_units
from rag.errors import RAGError
from rag.ingestion import document_hash, extract


class Library:
    def __init__(self, store, settings, service):
        self.store, self.settings, self.service = store, settings, service

    def ingest(self, name, raw, force=False):
        digest = document_hash(raw)
        existing = next((d for d in self.store.documents() if d["id"] == digest), None)
        if existing and not force:
            return {**existing, "duplicate": True}
        units, metadata, warnings = extract(name, raw, self.settings)
        chunks = chunk_units(units, digest, self.settings, self.service)
        if not chunks:
            raise RAGError("No searchable chunks were produced.")
        metadata["chunks"] = len(chunks)
        self.store.put_document(metadata, raw, units, chunks, self.settings.index_config(), warnings)
        return {**metadata, "id": digest, "warnings": warnings, "duplicate": False}

    def reindex(self, doc_id):
        return self.ingest(*self.store.raw_document(doc_id), force=True)
