"""Mature RAG path: Qwen-Agent components + Ollama (qwen2.5:7b).

Contrast with ../rag (the from-scratch dense+BM25 implementation kept as a baseline).
"""

from .qwen_rag import collect_files, parse_all

__all__ = ["collect_files", "parse_all"]
