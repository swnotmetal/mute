"""Minimal multi-backend LLM client.

Both backends expose the OpenAI-compatible /v1/chat/completions endpoint, so a
single code path serves them:
  * backend="ollama"   -> http://localhost:11434    (model e.g. "qwen2.5:7b")
  * backend="llamacpp" -> http://127.0.0.1:8080     (llama.cpp llama-server)

For llamacpp the loaded model id is auto-detected from /v1/models.
"""

from __future__ import annotations

import requests

OLLAMA_BASE = "http://localhost:11434"
LLAMACPP_BASE = "http://127.0.0.1:8080"

_cached_llamacpp_model: str | None = None


def llamacpp_model() -> str:
    global _cached_llamacpp_model
    if _cached_llamacpp_model is None:
        r = requests.get(f"{LLAMACPP_BASE}/v1/models", timeout=10)
        r.raise_for_status()
        _cached_llamacpp_model = r.json()["data"][0]["id"]
    return _cached_llamacpp_model


def base_url(backend: str) -> str:
    if backend == "ollama":
        return OLLAMA_BASE
    if backend == "llamacpp":
        return LLAMACPP_BASE
    raise ValueError(f"unknown backend: {backend!r}")


def generate(prompt: str, backend: str = "ollama", model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 160, timeout: int = 1800) -> str:
    """Single-turn generation via /v1/chat/completions."""
    if model is None:
        model = "qwen2.5:7b" if backend == "ollama" else llamacpp_model()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(f"{base_url(backend)}/v1/chat/completions", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from {backend} ({model}): {r.text[:600]}")
    return str(r.json()["choices"][0]["message"]["content"]).strip()
