"""Dense embeddings via Ollama /api/embed.

Uses bge-m3 (multilingual: Finnish / English / Chinese), so retrieval works
across a mixed-language corpus without per-language tokenisation tricks.
"""
from __future__ import annotations

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Batch-embed texts. Ollama /api/embed accepts a list of inputs."""
    if not texts:
        return []
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)
