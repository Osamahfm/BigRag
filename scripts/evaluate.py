"""Run from project root: python -m scripts.evaluate --backend demo."""
import argparse
import json
from pathlib import Path

from config import ROOT, Settings, data_directory
from rag.embeddings import DemoService, GeminiService
from rag.evaluation import compare, load_dataset
from rag.library import Library
from rag.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["demo", "gemini"], default="demo")
    parser.add_argument("--dataset", type=Path, default=ROOT / "examples/evaluation.json")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--load-examples", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--k", type=int, choices=range(1, 101), default=5)
    parser.add_argument("--model-judge", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation-results.json")
    args = parser.parse_args()
    settings = Settings(backend=args.backend)
    service = DemoService(settings) if args.backend == "demo" else GeminiService(settings)
    store = Store(args.data_dir or data_directory())
    if args.load_examples:
        library = Library(store, settings, service)
        for name in ("mapreduce.md", "spark.txt", "biology.txt"):
            library.ingest(name, (ROOT / "examples" / name).read_bytes())
    result = compare(store, settings, service, load_dataset(args.dataset),
                     k=args.k, generation=args.generation, model_judge=args.model_judge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved measured results to {args.output}")


if __name__ == "__main__":
    main()
