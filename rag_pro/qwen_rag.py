"""Mature RAG pipeline built on Qwen-Agent + Ollama (the "don't reinvent" path).

Components borrowed from Qwen-Agent (see its `qwen-agent[rag]` docs):
  - DocParser      : parses pdf/docx/pptx/xlsx/csv/html into chunked records
                     (handles docx tables that a naive `paragraphs` loader misses)
  - KeywordSearch  : BM25 sparse retrieval
  - SplitQueryThenGenKeyword : LLM decomposes the question + generates multilingual
                     keywords (helps multi-term / proper-noun queries)

We add one practical touch: the filename is prepended to each chunk so descriptive
names ("Wu_Shuang_Phishing-Email-Practise.docx") are searchable even when the word
never appears in the body.

Parsing is done ONCE and cached to a pickle; queries only do search + generate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import json5
import requests
from qwen_agent.agents.keygen_strategies.split_query_then_gen_keyword import (
    SplitQueryThenGenKeyword,
)
from qwen_agent.llm.schema import Message
from qwen_agent.tools.doc_parser import DocParser, Record
from qwen_agent.tools.search_tools.keyword_search import KeywordSearch

from .embed import EMBED_MODEL, cosine, embed
from .rerank import Reranker

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".md", ".csv", ".tsv", ".html"}
OLLAMA = "http://localhost:11434"
LLM_CFG = {
    "model": "qwen2.5:7b",
    "model_server": f"{OLLAMA}/v1",
    "api_key": "EMPTY",
    "generate_cfg": {"max_tokens": 512},
}


def collect_files(root: str) -> list[str]:
    r = Path(root)
    return [str(p) for p in sorted(r.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]


def _cache_path(root: str) -> str:
    h = hashlib.md5(str(Path(root).resolve()).encode()).hexdigest()[:10]
    return os.path.join(os.path.dirname(__file__), f".parse_cache_{h}.pkl")


def parse_all(files: list[str], cache_path: str, refresh: bool = False) -> list[dict]:
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    parser = DocParser({"max_ref_token": 20000, "parser_page_size": 500})
    records: list[dict] = []
    for i, f in enumerate(files, 1):
        try:
            rec = parser.call({"url": f})
        except Exception as e:
            print(f"[skip] {Path(f).name}: {type(e).__name__}: {e}")
            continue
        name = Path(f).name
        # Filename is part of the searchable text (the topic word often lives only in the
        # filename). Split on _ - . so Qwen-Agent's tokenizer sees "phishing" as its own
        # token instead of part of "wu_shuang_phishing-email-practise".
        clean_name = re.sub(r"[_\-.]", " ", Path(f).name)
        for pg in rec["raw"]:
            pg["content"] = f"[file: {clean_name}]\n" + (pg.get("content") or "")
        records.append(rec)
        if i % 20 == 0:
            print(f"  parsed {i}/{len(files)}")

    with open(cache_path, "wb") as f:
        pickle.dump(records, f)
    return records


def keywords_for(question: str, keygen: SplitQueryThenGenKeyword) -> str:
    *_, last = keygen.run([Message("user", question)])
    content = (last[-1].content or "").strip()
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            d = json5.loads(m.group(0))
            kws = list(d.get("keywords_en", [])) + list(d.get("keywords_zh", []))
            if kws:
                return ", ".join(kws)
        except Exception:
            pass
    return question


def ollama_answer(question: str, context: str) -> str:
    prompt = (
        "Answer the question using ONLY the context below. "
        'If the answer is not present, reply exactly "资料中没有相关信息" and do not guess.\n\n'
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False,
              "options": {"temperature": 0}},
        timeout=600,
    )
    r.raise_for_status()
    return str(r.json()["response"]).strip()


def retrieve(records: list[dict], query: str, k: int = 12) -> list[tuple[str, str, float]]:
    """Global BM25 top-k chunks: [(url, text, score), ...] with score > 0."""
    searcher = KeywordSearch({"max_ref_token": 20000})
    docs = [Record(**r) for r in records]
    ranked = searcher.sort_by_scores(query, docs)  # [(source, chunk_id, score)] sorted desc
    content_map = {(d.url, pg.metadata["chunk_id"]): pg.content for d in docs for pg in d.raw}
    out: list[tuple[str, str, float]] = []
    for source, cid, score in ranked:
        if score <= 0:
            continue
        text = content_map.get((source, cid))
        if text:
            out.append((source, text, float(score)))
        if len(out) >= k:
            break
    return out


def _chunks(records: list[dict]) -> tuple[list[tuple[str, str]], list[str]]:
    """Stable (key, text) lists over all chunks; key = (url, chunk_id)."""
    docs = [Record(**r) for r in records]
    keys: list[tuple[str, str]] = []
    texts: list[str] = []
    for d in docs:
        for pg in d.raw:
            keys.append((d.url, pg.metadata["chunk_id"]))
            texts.append(pg.content)
    return keys, texts


def chunk_vectors(records: list[dict]) -> tuple[list[tuple[str, str]], list[list[float]]]:
    """Embed every chunk once via bge-m3; cache by a hash of the chunk keys."""
    keys, texts = _chunks(records)
    h = hashlib.md5(str(keys).encode()).hexdigest()[:10]
    path = os.path.join(os.path.dirname(__file__), f".embed_cache_{h}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("model") == EMBED_MODEL and len(cached.get("vectors", [])) == len(keys):
            return keys, cached["vectors"]
    vectors: list[list[float]] = []
    batch = 16
    for i in range(0, len(texts), batch):
        vectors.extend(embed(texts[i:i + batch]))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}")
    with open(path, "wb") as f:
        pickle.dump({"keys": keys, "vectors": vectors, "model": EMBED_MODEL}, f)
    return keys, vectors


def retrieve_hybrid(records: list[dict], dense_query: str, bm25_query: str,
                    k: int = 12, rrf_k: int = 60, max_per_doc: int = 2) -> list[tuple[str, str, float]]:
    """BM25 (sparse keywords) + bge-m3 (dense, cross-lingual) fused by RRF.

    BM25 alone fails on a Finnish/English corpus with English questions; bge-m3
    matches meaning across languages, so the two are complementary.

    Dedup is implicit: results are keyed by (url, chunk_id) in a dict, so a chunk
    found by both retrievers appears once. `max_per_doc` caps how many chunks a
    single document may contribute, so one big file (e.g. a 200-page PDF) cannot
    monopolise the candidate pool and starve other documents of slots.
    """
    docs = [Record(**r) for r in records]
    text_map = {(d.url, pg.metadata["chunk_id"]): pg.content for d in docs for pg in d.raw}

    searcher = KeywordSearch({"max_ref_token": 20000})
    bm25 = [(u, cid) for u, cid, s in searcher.sort_by_scores(bm25_query, docs) if s > 0]

    keys, vectors = chunk_vectors(records)
    qv = embed([dense_query])[0]
    dense = [kk for kk, _ in sorted(
        ((keys[i], cosine(qv, vectors[i])) for i in range(len(keys))),
        key=lambda x: x[1], reverse=True)]

    fused: dict[tuple[str, str], float] = {}
    for lst in (bm25, dense):
        for rank, kk in enumerate(lst):
            fused[kk] = fused.get(kk, 0.0) + 1.0 / (rrf_k + rank + 1)

    out: list[tuple[str, str, float]] = []
    per_doc: dict[str, int] = {}
    for kk, sc in sorted(fused.items(), key=lambda x: x[1], reverse=True):
        if kk not in text_map:
            continue
        doc = kk[0]
        if per_doc.get(doc, 0) >= max_per_doc:
            continue
        out.append((kk[0], text_map[kk], sc))
        per_doc[doc] = per_doc.get(doc, 0) + 1
        if len(out) >= k:
            break
    return out


def answer_question(records: list[dict], question: str, retrieve_k: int = 12,
                    top_k: int = 5, rerank: bool = False, reranker: "Reranker | None" = None):
    """keyword-gen -> hybrid(BM25+dense RRF) top-k -> (optional) rerank -> answer."""
    keygen = SplitQueryThenGenKeyword(llm=LLM_CFG)
    query = keywords_for(question, keygen)
    cands = retrieve_hybrid(records, dense_query=question, bm25_query=query, k=retrieve_k)
    if rerank and cands:
        reranker = reranker or Reranker()
        kept = [c for c, _ in reranker.filter_topk(
            question, [(u, t) for u, t, _ in cands], top_k=top_k, threshold=0.0)]
    else:
        kept = [(u, t) for u, t, _ in cands[:top_k]]
    if not kept:
        return "资料中没有相关信息", query, []
    context = "\n\n".join(f"[{u}]\n{t}" for u, t in kept)
    return ollama_answer(question, context), query, kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen-Agent + Ollama RAG (keyword-gen -> BM25 top-k -> rerank -> answer)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--refresh", action="store_true", help="re-parse (ignore cache)")
    ap.add_argument("--retrieve-k", type=int, default=12, help="hybrid candidates before top-k")
    ap.add_argument("--top-k", type=int, default=5, help="chunks kept for the generator")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker")
    args = ap.parse_args()

    files = collect_files(args.dir)
    print(f"files found: {len(files)}")
    records = parse_all(files, _cache_path(args.dir), refresh=args.refresh)
    print(f"parsed records: {len(records)}")

    ans, query, kept = answer_question(records, args.question, retrieve_k=args.retrieve_k,
                                       top_k=args.top_k, rerank=not args.no_rerank)
    print(f"search keywords: {query}")
    print("=== KEPT ===")
    for u, t in kept:
        print(f"  {Path(u).name} :: {t[:60]!r}")
    print("\n=== ANSWER ===")
    print(ans)


if __name__ == "__main__":
    main()
