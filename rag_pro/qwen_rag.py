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
from qwen_agent.utils.tokenization_qwen import count_tokens

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


def _pypdf_text(path: str) -> str:
    """Plain text from a PDF via the installed pypdf (text layer only).

    shortcut: no OCR; a scanned PDF still yields nothing, see parse_all.
    """
    try:
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return ""


_DOCLING = None


def _docling_text(path: str) -> str:
    """Markdown via Docling: layout, tables and OCR. Slow on CPU.

    shortcut: one shared converter, CPU layout model, seconds per page. Use only
    as a fallback; upgrade to a GPU/batch pipeline if it becomes the main parser.
    """
    global _DOCLING
    try:
        from docling.document_converter import DocumentConverter

        if _DOCLING is None:
            _DOCLING = DocumentConverter()
        return _DOCLING.convert(path).document.export_to_markdown()
    except Exception:
        return ""


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
        # Scanned PDFs parse to near-empty chunks. Retry with pypdf's text layer;
        # if that is empty too, drop the document so it cannot occupy a context
        # slot as a silent blank chunk.
        # shortcut: no OCR - upgrade to an OCR parser if these files matter.
        if all(len((pg.get("content") or "").strip()) < 50 for pg in rec["raw"]):
            txt = _pypdf_text(f) if f.lower().endswith(".pdf") else ""
            if len(txt.strip()) < 50:
                txt = _docling_text(f)  # layout + tables + OCR; slow, last resort
            if len(txt.strip()) >= 50:
                rec["raw"] = [{"content": txt, "metadata": {"chunk_id": "fallback-0"},
                               "token": count_tokens(txt)}]
            else:
                print(f"[skip] {Path(f).name}: no extractable text (image only?)")
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


PROMPTS = {
    "grounded": (
        "Answer the question using ONLY the context below; do not use outside knowledge "
        "and do not invent details that are not in the context.\n"
        "- If the context describes or contains facts about the topic asked (a project, an "
        "assignment, a document about it), answer by summarizing what the context shows, "
        "even if it never literally says \"the student learned\".\n"
        '- Only reply exactly "资料中没有相关信息" when the context is entirely unrelated '
        "to the question."
    ),
    "helpful": (
        "You are a helpful, confident assistant. Answer the question directly using the "
        "information below, and make reasonable inferences where needed. Never reply that "
        "the information is missing or that you cannot answer - always give your best answer."
    ),
}


def ollama_answer(question: str, context: str, style: str = "grounded") -> str:
    instruction = PROMPTS.get(style, PROMPTS["grounded"])
    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
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
                    k: int = 12, rrf_k: int = 60, max_per_doc: int = 2,
                    expand_docs: int = 3, expand_per_doc: int = 3) -> list[tuple[str, str, float]]:
    """BM25 (sparse keywords) + bge-m3 (dense, cross-lingual) fused by RRF.

    BM25 alone fails on a Finnish/English corpus with English questions; bge-m3
    matches meaning across languages, so the two are complementary.

    Dedup is implicit: results are keyed by (url, chunk_id) in a dict, so a chunk
    found by both retrievers appears once. `max_per_doc` caps how many chunks a
    single document may contribute, so one big file cannot monopolise the pool.

    Document expansion (`expand_docs`/`expand_per_doc`): the fused ranking tends to
    surface only a document's FIRST chunk (title / table-of-contents page), which
    carries the filename but not the answer. For the top-scoring documents we also
    pull their chunks that are most semantically similar to the query, so body
    passages (e.g. a signature page, a "Created by X and Y" line) enter the pool.
    """
    docs = [Record(**r) for r in records]
    text_map = {(d.url, pg.metadata["chunk_id"]): pg.content for d in docs for pg in d.raw}

    searcher = KeywordSearch({"max_ref_token": 20000})
    bm25 = [(u, cid) for u, cid, s in searcher.sort_by_scores(bm25_query, docs) if s > 0]

    keys, vectors = chunk_vectors(records)
    qv = embed([dense_query])[0]
    sim = {keys[i]: cosine(qv, vectors[i]) for i in range(len(keys))}
    dense = [kk for kk, _ in sorted(sim.items(), key=lambda x: x[1], reverse=True)]

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

    # Document expansion: add the best-matching body chunks of the top documents.
    if expand_docs > 0 and expand_per_doc > 0:
        doc_best: dict[str, float] = {}
        for kk, sc in fused.items():
            doc_best[kk[0]] = max(doc_best.get(kk[0], 0.0), sc)
        top_docs = [d for d, _ in sorted(doc_best.items(), key=lambda x: -x[1])[:expand_docs]]
        have = {kk for kk, _, _ in out}
        for d in top_docs:
            extra = [kk for kk in sim if kk[0] == d and kk not in have]
            extra.sort(key=lambda kk: -sim[kk])
            for kk in extra[:expand_per_doc]:
                out.append((kk[0], text_map[kk], sim[kk]))
                have.add(kk)
    return out


def retrieve_context(records: list[dict], question: str, retrieve_k: int = 12,
                     top_k: int = 5, reranker: "Reranker | None" = None,
                     use_keywords: bool = True):
    """keyword-gen -> hybrid (BM25+dense RRF) -> optional rerank -> gated top-k.

    Returns (query, kept, scores, n_cands). Single source of truth for retrieval
    so `answer_question` and the probes stay in sync.

    `use_keywords=False` skips the LLM keyword generator and uses the raw
    question for BM25; that removes the dependence on an Ollama LLM, leaving
    Ollama needed only for bge-m3 embeddings.
    """
    if use_keywords:
        keygen = SplitQueryThenGenKeyword(llm=LLM_CFG)
        query = keywords_for(question, keygen)
    else:
        query = question
    cands = retrieve_hybrid(records, dense_query=question, bm25_query=query, k=retrieve_k)
    if not cands:
        return query, [], [], 0
    if reranker is None:
        kept = [(u, t) for u, t, _ in cands[:top_k]]
        scores = [s for _, _, s in cands[:top_k]]
    else:
        pairs = reranker.filter_topk(question, [(u, t) for u, t, _ in cands],
                                     top_k=top_k, threshold=0.0)
        kept = [c for c, _ in pairs]
        scores = [s for _, s in pairs]
    return query, kept, scores, len(cands)


def answer_question(records: list[dict], question: str, retrieve_k: int = 12,
                    top_k: int = 5, rerank: bool = False, reranker: "Reranker | None" = None,
                    trace: dict | None = None, style: str = "grounded"):
    """retrieve -> generate. `style` picks the grounding prompt ("grounded"/"helpful").

    If `trace` is given it is filled with retrieval diagnostics
    (n_cands / kept / scores / max_score / style).
    """
    reranker = (reranker or Reranker()) if rerank else None
    query, kept, scores, n_cands = retrieve_context(records, question, retrieve_k, top_k, reranker)
    if trace is not None:
        trace.update({"n_cands": n_cands, "kept": kept, "scores": scores,
                      "max_score": max(scores) if scores else 0.0, "style": style})
    if not kept:
        return "资料中没有相关信息", query, []
    context = "\n\n".join(f"[{u}]\n{t}" for u, t in kept)
    return ollama_answer(question, context, style=style), query, kept


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
