"""Character-based chunking with overlap. Simple and predictable for zh + en."""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
