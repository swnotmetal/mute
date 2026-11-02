"""Cross-encoder reranker + CRAG-style relevance filter.

Why: a small local model (7B) gets confused by long, noisy context. BM25 returns
many candidates; a cross-encoder scores (query, passage) jointly and is far more
accurate than lexical overlap. We then keep only the Top-K above a relevance
threshold (the "corrective" filter), so the generator sees a tight, relevant
context instead of a pile of documents.

Model: BAAI/bge-reranker-v2-m3 (multilingual: Finnish / English / Chinese,
~2.2GB). The earlier English-only bge-reranker-base mis-scored Finnish documents
(~0.0003 for the right doc, 0.43 for a wrong one), which is why the pipeline now
defaults to the multilingual v2-m3.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str | None = None, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 8) -> list[float]:
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
                    top_k: int = 3, threshold: float = 0.0,
                    ratio: float = 0.15) -> list[tuple[tuple[str, str], float]]:
        """candidates = [(url, text), ...] -> [((url, text), score), ...] filtered.

        Keeps at most top_k chunks whose score clears a *relative* floor
        `max(threshold, ratio * best_score)`. A relative floor adapts per query:
        when the reranker separates signal from noise (e.g. 0.98 vs 0.002) the
        noise is dropped, but when all candidates are weak-but-relevant
        (e.g. 0.025 / 0.010 / 0.008) they are all kept instead of being wiped out
        by a fixed absolute threshold.
        """
        if not candidates:
            return []
        scores = self.score(query, [c[1] for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        best = ranked[0][1]
        floor = max(threshold, ratio * best)
        return [(c, s) for c, s in ranked if s >= floor][:top_k]
