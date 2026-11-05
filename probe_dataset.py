"""Turn a mute_compare run into (context, question, label) rows for mute_hidden.

Why the context is rebuilt rather than stored: mute_compare.jsonl keeps the verdict
per field, not the text. The pipeline is frozen, so re-running retrieval reproduces
the same context; storing it would have meant keeping private document text in the
repo.

Two labels are written, and they are not the same thing:

  corpus_label   0 if the field is known to exist in the collection, else 1.
                 Hand-verified once, when the field list was built. It says
                 nothing about what the retriever actually handed the model.

  label          0 if the expected answer string is literally present in the
                 retrieved context, else 1. This is what the probe is asked to
                 predict: can the model answer from what it was given.

A control field whose evidence the retriever missed gets corpus_label 0 and
label 1. That is not a bug - it is exactly the "the corpus has it, the context
did not" case, and it is worth seeing how often it happens.
"""

from __future__ import annotations

import argparse
import json
import os

from mute_structured import FIELDS, build_context
from rag_pro.qwen_rag import _cache_path, collect_files, parse_all, retrieve_context
from rag_pro.rerank import Reranker


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a labelled probe dataset")
    ap.add_argument("--compare", default="probe_runs/compare.jsonl")
    ap.add_argument("--dir", required=True, help="corpus directory")
    ap.add_argument("--out", default="probe_runs/probe_dataset.jsonl")
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    fields = {k: (q, e) for k, q, e in FIELDS}
    rows = read_jsonl(args.compare)
    if not rows:
        raise SystemExit(f"{args.compare} is empty: run mute_compare.py first")

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()

    out: list[dict] = []
    for i, r in enumerate(rows, 1):
        key = r.get("key")
        if key not in fields:
            continue  # field was removed from FIELDS after this run
        question, expected = fields[key]
        _q, kept, scores, _n = retrieve_context(
            records, question, retrieve_k=args.retrieve_k, top_k=args.top_k,
            reranker=reranker, use_keywords=False)
        ctx = build_context(kept)
        found = expected is not None and expected.lower() in ctx.lower()
        out.append({
            "key": key,
            "question": question,
            "context": ctx,
            "label": 0 if found else 1,
            "corpus_label": 0 if expected is not None else 1,
            "expected": expected,
            "fabricated": r.get("fabricated"),
            "value": r.get("value"),
            "max_rr": r.get("max_rr"),
        })
        print(f"[{i}/{len(rows)}] {key:<24} label={out[-1]['label']} "
              f"corpus={out[-1]['corpus_label']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(out)
    pos = sum(1 for d in out if d["label"] == 0)
    lost = [d["key"] for d in out if d["corpus_label"] == 0 and d["label"] == 1]
    print(f"\nwrote {args.out}: n={n}  answerable={pos}  missing={n - pos}")
    print(f"controls whose evidence the retriever missed: {lost or 'none'}")
    minority = min(pos, n - pos)
    if minority < 10:
        print(f"WARNING: only {minority} samples in the minority class. An AUROC on "
              f"that is a coin flip with error bars; collect more before concluding.")


if __name__ == "__main__":
    main()
