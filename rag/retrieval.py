"""Embedding + cosine vector index (numpy only, no FAISS needed at this scale)."""

import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# bge-zh models expect this prefix on *queries* (not on documents).
_BGE_ZH_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """Mean/CLS-pooled, L2-normalized embeddings from a HuggingFace encoder."""

    def __init__(self, model_id: str, device: str | None = None, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.is_bge = "bge" in model_id.lower()
        self.is_zh = "zh" in model_id.lower()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        if is_query and self.is_bge and self.is_zh:
            texts = [_BGE_ZH_INSTRUCTION + t for t in texts]

        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            if self.is_bge:
                pooled = hidden[:, 0]  # CLS pooling (bge convention)
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


class VectorIndex:
    """Cosine similarity over pre-normalized vectors."""

    def __init__(self, vectors: np.ndarray, chunks: list):
        self.vectors = vectors
        self.chunks = chunks

    def search(self, query_vec: np.ndarray, k: int = 5):
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


class HybridIndex(VectorIndex):
    """Dense cosine + BM25 sparse, fused with Reciprocal Rank Fusion (RRF).

    BM25 rescues exact-keyword / proper-noun / multi-term queries that a single
    dense embedding averages away (e.g. "POWER-BI ... Samu").
    """

    def __init__(self, vectors: np.ndarray, chunks: list):
        super().__init__(vectors, chunks)
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])

    def search(self, query_vec: np.ndarray, query_text: str = "", k: int = 5, rrf_k: int = 60):
        dense_order = np.argsort(-(self.vectors @ query_vec))
        sparse_order = np.argsort(-self._bm25.get_scores(_tok(query_text)))

        fused: dict[int, float] = {}
        for order in (dense_order, sparse_order):
            for rank, i in enumerate(order):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], float(score)) for i, score in top]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    # Split on non-alphanumeric so "phishing?" == "wu_shuang_phishing-email-practise.docx".
    return _TOKEN_RE.findall(text.lower())
