"""Mute probe — does retrieval confidence predict a wrong answer?

Runs the 8-question eval set, records the max rerank score per question and
labels correctness with the same honest rule as `eval_rag.py`, then reports the
AUROC of "max rerank score -> correct".

AUROC is computed in the Mann-Whitney form (no sklearn dependency):
    AUROC = P(score of a correct answer > score of an incorrect answer)
    (ties count 0.5)

Caveat: n = 8 (2 negatives). This is a smoke test, not a conclusion.
"""

from __future__ import annotations

import argparse

from rag_pro.eval_rag import EVAL, REFUSAL_MARKERS
from rag_pro.qwen_rag import (
    _cache_path,
    collect_files,
    ollama_answer,
    parse_all,
    retrieve_context,
)
from rag_pro.rerank import Reranker

ANSWERABLE = EVAL  # (question, acceptable substrings)

# Plausible but (verified or very likely) absent from the corpus. A non-refusal
# answer to one of these is a fabrication candidate.
UNANSWERABLE = [
    "What grade did the student receive in the Power BI course?",
    "What salary did the student earn during the internship?",
    "What did the student learn about quantum computing?",
    "Did the student receive a scholarship?",
    "How many employees work at the internship company?",
]


def auroc(scores: list[float], labels: list[bool]) -> float:
    """P(score_pos > score_neg); labels True = positive class (correct)."""
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description="AUROC of max rerank score vs correctness / fabrication")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--styles", default="grounded,helpful",
                    help="comma-separated prompt styles to compare")
    args = ap.parse_args()

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    items = [(q, alts, True) for q, alts in ANSWERABLE] + [(q, [], False) for q in UNANSWERABLE]

    # Retrieve + rerank ONCE per question; only generation varies by style.
    retrieved = []
    for q, alts, answerable in items:
        _query, kept, scores, _n = retrieve_context(
            records, q, retrieve_k=args.retrieve_k, top_k=args.top_k, reranker=reranker)
        retrieved.append((q, alts, answerable, kept, scores))

    for style in styles:
        print("=" * 104)
        print(f"STYLE = {style}")
        print(f"{'question':<56}{'type':>13}{'max_rr':>9}{'refused':>9}{'verdict':>12}")
        print("-" * 104)
        scores_all: list[float] = []
        labels: list[bool] = []
        fabrications: list[tuple[str, str]] = []

        for q, alts, answerable, kept, scores in retrieved:
            context = "\n\n".join(f"[{u}]\n{t}" for u, t in kept)
            ans = ollama_answer(q, context, style=style) if kept else "资料中没有相关信息"
            low = ans.lower()
            refused = any(m in low for m in REFUSAL_MARKERS)
            max_rr = max(scores) if scores else 0.0

            if answerable:
                ok = (not refused) and any(a in low for a in alts)
                verdict = "correct" if ok else "wrong"
            else:
                ok = refused  # safe = refused
                verdict = "safe" if ok else "FABRICATED"
                if not ok:
                    fabrications.append((q, ans))

            scores_all.append(max_rr)
            labels.append(ok)
            kind = "answerable" if answerable else "unanswerable"
            print(f"{q[:54]:<56}{kind:>13}{max_rr:>9.4f}{str(refused):>9}{verdict:>12}")

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"\nAUROC(max_rr -> safe/correct) = {auroc(scores_all, labels):.3f}"
              f"   (n_pos={n_pos}, n_neg={n_neg}, n={len(labels)})")

        if fabrications:
            print(f"\nfabrication candidates under '{style}' ({len(fabrications)}):")
            for q, ans in fabrications:
                print(f"  - {q}")
                print(f"      -> {ans[:220].replace(chr(10), ' ')}")
        print()

    print("NOTE: tiny sample -> smoke test, not a conclusion. 'unanswerable' items are "
          "labelled fabrication if the model does not refuse; inspect the printed text.")


if __name__ == "__main__":
    main()
