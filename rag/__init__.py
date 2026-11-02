"""Minimal, dependency-light RAG package for the Mute project.

Phase 1 (this): standard dense RAG — load docs -> chunk -> embed -> cosine retrieve
-> Qwen generate.  Phase 2 will add BM25 hybrid + BGE reranker; phase 3 wires the
Mute probe in to detect retrieval failure (missing context) as the hallucination
experiment.
"""

from .config import RagConfig

__all__ = ["RagConfig"]
