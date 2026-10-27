"""Bounded in-process ingestion worker with durable job history.

Uploads remain in memory until extracted. Interrupted work is explicitly marked
on restart and must be submitted again; this is not a distributed task queue.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from rag.embeddings import DemoService, GeminiService
from rag.errors import RAGError
from rag.ingestion import validate_upload
from rag.library import Library


class IngestionJobs:
    def __init__(self, store, service_factory=None):
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-ingestion")
        self.lock = threading.RLock()
        self.futures = {}
        self.service_factory = service_factory or self._service
        with store.connect() as db:
            db.execute("UPDATE jobs SET status='interrupted',payload=? WHERE status IN ('queued','running')",
                       ('{"message":"Application restarted. Submit this upload again."}',))

    @staticmethod
    def _service(settings):
        if settings.chunk_strategy != "semantic":
            return None
        return DemoService(settings) if settings.backend == "demo" else GeminiService(settings)

    def submit(self, filename, raw, settings):
        filename, _ = validate_upload(filename, raw, settings)
        with self.lock:
            self.futures = {identifier: future for identifier, future in self.futures.items() if not future.done()}
            if len(self.futures) >= 10:
                raise RAGError("Ingestion queue is full. Wait for existing jobs to finish.")
            identifier = self.store.new_job(filename, {"bytes": len(raw), "settings": asdict(settings)})
            self.futures[identifier] = self.executor.submit(self._run, identifier, filename, raw, settings)
            return identifier

    def _run(self, identifier, filename, raw, settings):
        self.store.update_job(identifier, "running", 0.2)
        try:
            service = self.service_factory(settings)
            if hasattr(service, "error_observer"):
                service.error_observer = lambda event: self.store.event("api_error", event)
            result = Library(self.store, settings, service).ingest(filename, raw)
            self.store.update_job(identifier, "completed", 1, result)
            self.store.event("ingestion", {"source": filename, "chunks": result["chunks"], "job_id": identifier})
        except RAGError as exc:
            self.store.update_job(identifier, "failed", 1, {"message": str(exc)})
            self.store.event("ingestion_error", {"source": filename, "message": str(exc)})
        except Exception:
            self.store.update_job(identifier, "failed", 1, {"message": "Document processing failed. Check format dependencies."})

    def cancel(self, identifier):
        with self.lock:
            future = self.futures.get(identifier)
            if future and future.cancel():
                self.store.update_job(identifier, "cancelled", 0)
                return True
            return False

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
