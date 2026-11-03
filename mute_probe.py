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
from rag_pro.qwen_rag import _cache_path, answer_question, collect_files, parse_all
from rag_pro.rerank import Reranker


def auroc(scores: list[float], labels: list[bool]) -> float:
    """P(score_pos > score_neg); labels True = positive class (correct)."""
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description="AUROC of max rerank score vs correctness")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()

    rows = []
    for q, alts in EVAL:
        trace: dict = {}
        ans, _kw, _kept = answer_question(
            records, q, retrieve_k=args.retrieve_k, top_k=args.top_k,
            rerank=True, reranker=reranker, trace=trace)
        low = ans.lower()
        refused = any(m in low for m in REFUSAL_MARKERS)
        correct = (not refused) and any(a in low for a in alts)
        rows.append((q, float(trace.get("max_score", 0.0)), len(trace.get("kept", [])),
                     refused, correct))

    print(f"{'question':<58}{'max_rr':>9}{'kept':>6}{'refused':>9}{'correct':>9}")
    print("-" * 91)
    for q, ms, nk, rf, ok in rows:
        print(f"{q[:56]:<58}{ms:>9.4f}{nk:>6}{str(rf):>9}{str(ok):>9}")

    scores = [r[1] for r in rows]
    labels = [r[4] for r in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    auc_correct = auroc(scores, labels)
    print(f"\nn_correct={n_pos}  n_incorrect={n_neg}  (n={len(rows)})")
    print(f"correct max_scores    : {[round(s, 4) for s, l in zip(scores, labels) if l]}")
    print(f"incorrect max_scores  : {[round(s, 4) for s, l in zip(scores, labels) if not l]}")
    print(f"\nAUROC(max_rr -> correct)   = {auc_correct:.3f}")
    print(f"AUROC(max_rr -> incorrect) = {1 - auc_correct:.3f}   (use this to flag failures)")
    print("\nNOTE: n=8 with only 2 negatives -> wide uncertainty. Smoke test, not a conclusion.")


if __name__ == "__main__":
    main()
