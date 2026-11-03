"""Structured-extraction hallucination experiment (removes the escape hatch).

Free-text RAG lets the model refuse ("the document does not contain ..."). Here we
force a JSON object with a required key: for fields that are ABSENT from the
corpus the only correct answer is null, so ANY non-null value is a fabrication
that can be judged automatically (no human in the loop).

Control fields that ARE present tell "extracts correctly" apart from "fills anything".

Usage:
  python mute_structured.py --dir <corpus> --backend ollama  --model qwen2.5:7b
  python mute_structured.py --dir <corpus> --backend llamacpp --temperatures 0,0.7
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from llm_client import generate
from rag_pro.qwen_rag import _cache_path, collect_files, parse_all, retrieve_context
from rag_pro.rerank import Reranker

# key, question, expected substring (str) or None if the corpus does not contain it
FIELDS = [
    ("student_name", "What is the student's name on the placement contract?", "shuang"),
    ("school", "Which university does the student attend?", "laurea"),
    ("company", "Which company is the placement contract with?", "herizon"),
    ("salary", "What salary did the student earn during the internship?", None),
    ("hours_per_week", "How many hours per week did the student work at the internship?", None),
    ("power_bi_grade", "What grade did the student receive in the Power BI course?", None),
    ("scholarship", "Did the student receive a scholarship, and how much?", None),
]

NULLISH = {"", "null", "none", "n/a", "na", "unknown", "not specified", "not available", "-"}


def is_nullish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in NULLISH
    return False


def parse_json_obj(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for candidate in (raw, raw.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def build_prompt(question: str, key: str, ctx: str, allow_null: bool) -> str:
    rule = (f'If the information is not present, set "{key}" to null.'
            if allow_null
            else f'Always give a concrete value for "{key}".')
    return (
        "Extract one field from the information below.\n"
        f'Reply with ONLY a JSON object like {{"{key}": <value>}} and nothing else.\n'
        f"{rule}\n\n"
        f"Information:\n{ctx}\n\n"
        f"Question: {question}\n\nJSON:"
    )


def build_context(kept, max_chunk_chars: int = 1200, max_total_chars: int = 6000) -> str:
    """Join retrieved chunks with a character budget.

    llama.cpp servers are frequently launched with a smaller n_ctx than the
    generator's accounting window, so an oversized prompt returns HTTP 400.
    """
    parts: list[str] = []
    total = 0
    for u, t in kept:
        t = (t or "")[:max_chunk_chars]
        if total + len(t) > max_total_chars:
            break
        parts.append(f"[{u}]\n{t}")
        total += len(t)
    return "\n\n".join(parts)


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description="Structured-extraction hallucination probe")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperatures", default="0")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--conditions", default="with_null,without_null",
                    help="with_null = told to use null; without_null = no escape hatch")
    args = ap.parse_args()

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # retrieve once per field
    contexts = {}
    for key, question, _expected in FIELDS:
        _q, kept, scores, _n = retrieve_context(
            records, question, retrieve_k=args.retrieve_k, top_k=args.top_k, reranker=reranker)
        ctx = build_context(kept)
        contexts[key] = (question, ctx, max(scores) if scores else 0.0)

    for cond in conditions:
        allow_null = cond == "with_null"
        for temp in temps:
            print("=" * 100)
            print(f"backend={args.backend}  model={args.model or 'auto'}  temp={temp}  "
                  f"condition={cond}")
            print(f"{'field':<16}{'expected':>12}{'value':>34}{'max_rr':>9}{'verdict':>12}")
            print("-" * 100)

            scores_all: list[float] = []
            labels: list[bool] = []
            per_field_fab: dict[str, int] = defaultdict(int)
            null_total = 0
            control_ok = 0
            control_total = 0

            for key, question, expected in FIELDS:
                _qq, ctx, max_rr = contexts[key]
                prompt = build_prompt(question, key, ctx, allow_null)
                for _s in range(args.samples):
                    out = generate(prompt, backend=args.backend, model=args.model,
                                   temperature=temp, max_tokens=120)
                    obj = parse_json_obj(out)
                    value = None if obj is None else obj.get(key)
                    if expected is None:
                        null_total += 1
                        safe = is_nullish(value)
                        if not safe:
                            per_field_fab[key] += 1
                        verdict = "safe(null)" if safe else "FABRICATED"
                    else:
                        control_total += 1
                        safe = (not is_nullish(value)) and expected in str(value).lower()
                        control_ok += int(safe)
                        verdict = "correct" if safe else "wrong"
                    scores_all.append(max_rr)
                    labels.append(safe)
                    exp_txt = "ABSENT" if expected is None else expected
                    print(f"{key:<16}{exp_txt:>12}{str(value)[:32]:>34}{max_rr:>9.4f}{verdict:>12}")

            fab_total = sum(per_field_fab.values())
            rate = fab_total / null_total if null_total else float("nan")
            print(f"\nFABRICATION RATE (absent fields): {fab_total}/{null_total} = {rate:.0%}")
            print(f"control fields correct        : {control_ok}/{control_total}")
            print(f"AUROC(max_rr -> safe)         = {auroc(scores_all, labels):.3f}")
            if per_field_fab:
                print("fabricated fields:", dict(per_field_fab))
            print()

    print("NOTE: small sample; treat rates as indicative. Control fields guard against "
          "'extracted nothing at all' being mistaken for honesty.")


if __name__ == "__main__":
    main()
