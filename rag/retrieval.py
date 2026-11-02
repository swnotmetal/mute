"""Embedding + cosine vector index (numpy only, no FAISS needed at this scale)."""

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
