import os
import time

import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import ROOT
from rag.errors import RAGError


def api_call(operation, sleep=time.sleep, on_error=None):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            code = getattr(exc, "code", None)
            if on_error:
                on_error({"code": code if isinstance(code, int) else None, "attempt": attempt+1})
            transient = code in (429, 500, 502, 503, 504) or isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
            if transient and attempt < 2:
                sleep(2 ** attempt)
                continue
            if code in (400, 401, 403):
                message = "Gemini rejected the request. Check your API key, permissions, and model configuration."
            elif code == 404:
                message = "The configured Gemini model is unavailable for this API key."
            elif code == 429:
                message = "Gemini rate limit reached. Wait before trying again and check your quota."
            elif transient:
                message = "Gemini is temporarily unavailable or the network failed. Please try again later."
            else:
                message = "Gemini could not complete the request. Check your connection and model configuration."
            raise RAGError(message) from None


class GeminiService:
    def __init__(self, settings, client=None, api_key=None):
        self.settings = settings
        if client is None:
            load_dotenv(ROOT / ".env")
            key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
            if not key or not key.strip():
                raise RAGError("GEMINI_API_KEY is missing. Add it to the project's .env file and restart.")
            client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client
        self.usage_tokens = 0
        self.api_errors = 0
        self.error_observer = None

    def _record_error(self, event):
        self.api_errors += 1
        if self.error_observer:
            self.error_observer(event)

    def embed(self, texts, task_type):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            result = api_call(lambda batch=batch: self.client.models.embed_content(
                model=self.settings.embedding_model, contents=batch,
                config=types.EmbedContentConfig(task_type=task_type,
                                                output_dimensionality=self.settings.dimension)), on_error=self._record_error)
            values = [item.values for item in (result.embeddings or [])]
            try:
                array = np.asarray(values, dtype=np.float32)
            except (ValueError, TypeError):
                raise RAGError("Gemini returned invalid embeddings. Please retry indexing.") from None
            if array.shape != (len(batch), self.settings.dimension) or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
                raise RAGError("Gemini returned invalid embeddings. Please retry indexing.")
            vectors.extend(values)
        return np.asarray(vectors, dtype=np.float32)

    def generate(self, prompt, instruction):
        response = api_call(lambda: self.client.models.generate_content(
            model=self.settings.generation_model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=instruction, temperature=0.1)), on_error=self._record_error)
        usage = getattr(response, "usage_metadata", None)
        self.usage_tokens += getattr(usage, "total_token_count", 0) or 0
        return (response.text or "").strip()


class DemoService:
    """Deterministic hash vectors + extractive answers, explicitly NOT semantic AI."""
    def __init__(self, settings):
        self.settings = settings
        self.usage_tokens = 0

    def embed(self, texts, task_type):
        import hashlib

        from rag.text import tokens
        matrix = np.zeros((len(texts), self.settings.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self.settings.dimension] += 1
            if not matrix[row].any():
                matrix[row, 0] = 1
        return matrix

    def generate(self, prompt, instruction):
        import json
        import re
        payload = json.loads(prompt)
        if "history" in payload:
            previous = next((m["content"] for m in reversed(payload["history"]) if m["role"] == "user"), "")
            return payload["question"] + " (" + previous[:250] + ")"
        if payload.get("operation") == "expand":
            return json.dumps([payload["question"], payload["question"] + " explanation examples"])
        if payload.get("operation") == "verify":
            return json.dumps({"supported": True, "reason": "Extractive demo quotes only; no semantic judge used."})
        passages = payload.get("passages", [])
        if not passages:
            return "I couldn't find this information in the selected documents."
        query_words = set(__import__("rag.text", fromlist=["tokens"]).tokens(payload["standalone_question"]))
        answers = []
        for source in passages[:3]:
            sentences = re.split(r"(?<=[.!?؟])\s+|\n", source["text"])
            ranked = sorted(sentences, key=lambda s: len(query_words.intersection(
                __import__("rag.text", fromlist=["tokens"]).tokens(s))), reverse=True)
            excerpt = next((s.strip() for s in ranked if s.strip()), "")[:800]
            if excerpt:
                answers.append(excerpt + " " + source["citation"])
        return "\n\n".join(answers)
