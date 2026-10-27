import io
import json
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from config import Settings
from rag.embeddings import DemoService
from rag.library import Library
from rag.storage import Store


def make_pdf(texts, password=None):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        if text:
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def settings():
    return Settings(backend="demo", dimension=64, chunk_size=200, chunk_overlap=20, parent_size=600)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def service(settings):
    service = DemoService(settings)
    service.embed = Mock(wraps=service.embed)
    service.generate = Mock(wraps=service.generate)
    return service


@pytest.fixture
def corpus(store, settings, service):
    library = Library(store, settings, service)
    docs = [library.ingest("lecture.pdf", make_pdf(["MapReduce processes data in parallel.", "", "MapReduce advantages include scalability and fault tolerance."])),
            library.ingest("biology.txt", b"Photosynthesis uses sunlight to make sugar in plants."),
            library.ingest("spark.md", b"# Spark\nSpark caches intermediate data in memory for iterative algorithms.")]
    return store, docs


def fake_generation(prompt, instruction):
    payload = json.loads(prompt)
    if "history" in payload:
        return "What are the advantages of MapReduce?"
    if payload.get("operation") == "expand":
        return json.dumps([payload["question"], "MapReduce benefits fault tolerance"])
    if payload.get("operation") == "verify":
        return '{"supported": true, "reason": "supported by passage"}'
    return "MapReduce processes data in parallel. " + payload["passages"][0]["citation"]
