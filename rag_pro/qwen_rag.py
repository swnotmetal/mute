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


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen-Agent + Ollama RAG")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--refresh", action="store_true", help="re-parse (ignore cache)")
    ap.add_argument("--max-ref-token", type=int, default=4000, help="context budget (keep small for 7B)")
    args = ap.parse_args()

    files = collect_files(args.dir)
    print(f"files found: {len(files)}")
    records = parse_all(files, _cache_path(args.dir), refresh=args.refresh)
    print(f"parsed records: {len(records)}")

    keygen = SplitQueryThenGenKeyword(llm=LLM_CFG)
    query = keywords_for(args.question, keygen)
    print(f"search keywords: {query}")

    searcher = KeywordSearch({"max_ref_token": args.max_ref_token})
    hits = searcher.call({"query": query}, docs=[Record(**r) for r in records])
    context = "\n\n".join(f"[{h['url']}]\n" + "\n".join(h["text"]) for h in hits)

    print("\n=== ANSWER ===")
    print(ollama_answer(args.question, context))
    print("\n=== SOURCES ===")
    for h in hits:
        print(" ", h["url"])


if __name__ == "__main__":
    main()
