"""Optional real Gemini smoke test. Never prints credentials or raw API exceptions."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from config import ROOT, Settings
from rag.embeddings import GeminiService
from rag.errors import RAGError
from rag.library import Library
from rag.pipeline import Pipeline
from rag.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, default=ROOT / "data/live-smoke.json")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY") or dotenv_values(args.env_file).get("GEMINI_API_KEY")
    settings = Settings()
    result = {"backend": "gemini", "generation_model": settings.generation_model,
              "embedding_model": settings.embedding_model}
    try:
        service = GeminiService(settings, api_key=key or "")
        with tempfile.TemporaryDirectory(prefix="atlas-smoke-") as folder:
            store = Store(folder)
            Library(store, settings, service).ingest("smoke.txt", b"MapReduce processes data in parallel. Its advantages are scalability and fault tolerance.")
            answer = Pipeline(store, settings, service).ask("What are the advantages of MapReduce?", expansion=False)
            result.update({"success": bool(answer["cited_sources"]), "answer": answer["content"],
                           "metrics": answer["metrics"], "warnings": answer["debug"]["warnings"]})
    except RAGError as exc:
        result.update({"success": False, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
