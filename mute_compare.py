"""Controlled comparison of RAG abstention signals (the "honest" experiment).

Ground truth is automatic: run structured extraction WITHOUT the null escape
hatch. For a field the corpus does not contain, a non-null value = fabrication
(the model should have abstained).

Signals compared (all cheap, no hidden states):
  A. retrieval score   max rerank score  (low -> abstain)
  B. LLM self-eval     "can the context answer this?" YES/NO
  C. span attribution  does the extracted value appear in the context?
  D. MUTE probe        NOT implemented here (needs the CUAD hidden-state probe);
                       this script is the baseline ladder it must beat.

Output: per-field table + AUROC of each signal for predicting fabrication.
Caveat: small N; treat as a pilot.
"""

from __future__ import annotations

import argparse

from llm_client import generate
from mute_structured import (
    FIELDS,
    build_context,
    build_prompt,
    is_nullish,
    parse_json_obj,
    value_in_context,
)
from rag_pro.qwen_rag import _cache_path, collect_files, parse_all, retrieve_context
from rag_pro.rerank import Reranker

SELFEVAL = (
    "Information:\n{ctx}\n\nQuestion: {q}\n\n"
    "Can the question be answered using ONLY the information above? "
    "Reply with exactly one word: YES or NO."
)


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare abstention signals against fabrication GT")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()

    rows = []
    for key, question, expected in FIELDS:
        if expected is not None:
            continue  # abstention comparison only makes sense for absent fields
        _q, kept, scores, _n = retrieve_context(
            records, question, retrieve_k=args.retrieve_k, top_k=args.top_k, reranker=reranker)
        ctx = build_context(kept)
        max_rr = max(scores) if scores else 0.0

        out = generate(build_prompt(question, key, ctx, allow_null=False),
                       backend=args.backend, model=args.model,
                       temperature=args.temperature, max_tokens=120)
        obj = parse_json_obj(out)
        value = None if obj is None else obj.get(key)
        fabricated = not is_nullish(value)

        se = generate(SELFEVAL.format(ctx=ctx, q=question),
                      backend=args.backend, model=args.model,
                      temperature=0.0, max_tokens=8)
        self_no = se.strip().upper().startswith("NO")

        ungrounded = (not is_nullish(value)) and (not value_in_context(value, ctx))
        rows.append((key, value, fabricated, max_rr, self_no, ungrounded))

    print(f"{'field':<18}{'value':>22}{'fabricated':>11}{'max_rr':>9}{'self=NO':>9}{'unground':>10}")
    print("-" * 79)
    for key, value, fab, mr, se_no, ung in rows:
        print(f"{key:<18}{str(value)[:20]:>22}{str(fab):>11}{mr:>9.4f}{str(se_no):>9}{str(ung):>10}")

    labels = [r[2] for r in rows]
    n_fab = sum(labels)
    print(f"\nfabricated={n_fab}/{len(rows)}")

    # each signal scored so that HIGHER = more likely fabricated
    sig_retrieval = [r[3] for r in rows]          # tested in both directions below
    sig_selfeval = [1.0 if r[4] else 0.0 for r in rows]
    sig_span = [1.0 if r[5] else 0.0 for r in rows]

    print("\nAUROC for predicting FABRICATION (higher=better):")
    print(f"  A. retrieval  max_rr          = {auroc(sig_retrieval, labels):.3f}")
    print(f"  A'. retrieval  -max_rr        = {auroc([-s for s in sig_retrieval], labels):.3f}")
    print(f"  B. LLM self-eval says NO      = {auroc(sig_selfeval, labels):.3f}")
    print(f"  C. span attribution ungrounded= {auroc(sig_span, labels):.3f}")
    print("  D. MUTE hidden-state probe    = n/a (not wired yet)")

    print("\nNOTE: pilot with few absent fields; AUROC on tiny N is unstable. "
          "Values flagged 'unground=False' but fabricated=True mean the fake value "
          "happened to appear in context (label audit needed).")


if __name__ == "__main__":
    main()
