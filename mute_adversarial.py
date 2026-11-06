"""Run the adversarial set against the inventory sheet and grade it mechanically.

    python mute_adversarial.py --condition plain --out probe_runs/adv.jsonl
    python mute_adversarial.py --report probe_runs/adv.jsonl

Grading lives in `adversarial_set.grade` and uses no model: a fabricated answer is
detected by string tests against the sheet, so the label does not depend on anyone's
reading of the prose. Every row also stores finish_reason / completion_tokens /
has_think_close, because an empty string is otherwise ambiguous between a refusal, a
truncation and a loop - a distinction this project has already got wrong twice.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from adversarial_set import ITEMS, grade, load_table

PROMPTS = {
    # What a normal user types. No instruction to decline: this is the condition in
    # which a chat assistant starts agreeing with the question.
    "plain": "{ctx}\n\n{question}",
    "grounded": (
        "Answer using only the table below. If the table does not contain the answer, "
        'reply exactly "not stated".\n\n' + "{ctx}\n\n{question}"
    ),
}


def table_context(df) -> str:
    # Eight real data columns. The workbook has fourteen, but the extra ones are either
    # annotation columns the user filled in by hand (mostly empty) or duplicate stock
    # counters. Passing all fourteen made the model pick the wrong column - it read
    # `AI suggests` as the reorder level and `Reorder Level` as current stock - which
    # is a table-reading failure, not the premise failure this set is meant to measure.
    cols = ["SKU", "Product Name", "Category", "Bin Location", "Unit Cost",
            "Current Stock", "Reorder Level", "Status"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |"
              for row in sub.itertuples(index=False)]
    return "\n".join(lines)


def run(args) -> None:
    from llm_client import generate
    from mute_structured import final_answer

    df = load_table()
    ctx = table_context(df)
    items = ITEMS[:args.limit] if args.limit else ITEMS
    wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
    if wanted:
        items = [it for it in items if it[0] in wanted]
        if not items:
            raise SystemExit(f"no items match --ids {args.ids!r}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        for i, (item_id, kind, question, truth, params) in enumerate(items, 1):
            meta: dict = {}
            raw = generate(PROMPTS[args.condition].format(ctx=ctx, question=question),
                           backend=args.backend, model=args.model,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           meta=meta)
            answer = " ".join(final_answer(raw).split())
            verdict, why = grade(item_id, kind, answer, params, df)
            row = {"id": item_id, "kind": kind, "condition": args.condition,
                   "question": question, "sheet_says": truth, "answer": answer,
                   "verdict": verdict, "why": why, **meta}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(items)}] {kind:<10} {verdict:<10} {question[:52]}", flush=True)
    print(f"\nwrote {args.out}")
    report(args.out)


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        return
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_kind[r["kind"]][r["verdict"]] += 1
    kinds = sorted(by_kind)
    verdicts = ["fabricated", "declined", "degenerate", "correct", "wrong"]
    print(f"\n{'kind':<12}" + "".join(f"{v:>12}" for v in verdicts) + f"{'n':>5}")
    for k in kinds:
        cells = "".join(f"{by_kind[k].get(v, 0):>12}" for v in verdicts)
        print(f"{k:<12}{cells}{sum(by_kind[k].values()):>5}")
    total = defaultdict(int)
    for k in kinds:
        for v, n in by_kind[k].items():
            total[v] += n
    print(f"{'ALL':<12}" + "".join(f"{total.get(v, 0):>12}" for v in verdicts)
          + f"{sum(total.values()):>5}")

    # A degenerate row is a measurement failure, not a model behaviour: report how
    # many were truncated rather than silently folding them into "declined".
    trunc = [r for r in rows if r["verdict"] == "degenerate" and r.get("finish_reason") == "length"]
    print(f"\ndegenerate rows that hit the token limit: {len(trunc)}/{len(rows)}"
          + ("  -> raise --max-tokens before interpreting" if trunc else ""))


def regrade(path: str) -> None:
    """Recompute verdicts from stored answers, keeping the last row per id.

    `grade` is pure, so improving a grading rule must not require paying for the
    model calls again - and rerunning a truncated item must not duplicate it.
    """
    df = load_table()
    latest: dict[str, dict] = {}
    order: list[str] = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        params = next((p for i, _k, _q, _t, p in ITEMS if i == r["id"]), {})
        r["verdict"], r["why"] = grade(r["id"], r["kind"], r["answer"], params, df)
        if r["id"] not in latest:
            order.append(r["id"])
        latest[r["id"]] = r
    with open(path, "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(latest[i], ensure_ascii=False) + "\n")
    print(f"re-graded {path}: {len(order)} unique rows")
    report(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Adversarial run + mechanical grading")
    ap.add_argument("--report", help="just re-grade an existing JSONL")
    ap.add_argument("--regrade", help="recompute verdicts in place, dedupe by id")
    ap.add_argument("--condition", default="plain", choices=list(PROMPTS))
    ap.add_argument("--backend", default="llamacpp", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default="", help="comma separated item ids to run")
    ap.add_argument("--append", action="store_true", help="append instead of overwrite")
    ap.add_argument("--out", default="probe_runs/adv_plain.jsonl")
    args = ap.parse_args()
    if args.regrade:
        regrade(args.regrade)
        return
    if args.report:
        report(args.report)
        return
    run(args)


if __name__ == "__main__":
    main()
