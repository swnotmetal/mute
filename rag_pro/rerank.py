"""Cross-encoder reranker + CRAG-style relevance filter.

Why: a small local model (7B) gets confused by long, noisy context. BM25 returns
many candidates; a cross-encoder scores (query, passage) jointly and is far more
accurate than lexical overlap. We then keep only the Top-K above a relevance
threshold (the "corrective" filter), so the generator sees a tight, relevant
context instead of a pile of documents.

Model: BAAI/bge-reranker-base (English, ~278MB). For Chinese queries use
BAAI/bge-reranker-v2-m3 (multilingual, larger).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-base", device: str | None = None, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        if not passages:
            return []
        scores: list[float] = []
        for i in range(0, len(passages), batch_size):
            batch = [[query, p] for p in passages[i:i + batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.view(-1)
            scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        return scores

    def filter_topk(self, query: str, candidates: list[tuple[str, str]],
                    top_k: int = 3, threshold: float = 0.25) -> list[tuple[tuple[str, str], float]]:
        """candidates = [(url, text), ...] -> [((url, text), score), ...] filtered.

        Keeps only chunks with score >= threshold, at most top_k. If nothing passes,
        returns [] (CRAG: treat as "no relevant context").
        """
        if not candidates:
            return []
        scores = self.score(query, [c[1] for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked if s >= threshold][:top_k]
