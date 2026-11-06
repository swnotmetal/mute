"""Score the adversarial rows with cheap signals, then compare them on AUROC.

Three signals, matching the ones the project already uses, plus a fourth that is new
here and is the one that matters for premise violations:

  hhem      a 0.1B entailment model: does the answer get support from the table?
            Expected to FAIL on this set, because a fabricated rank or a padded list
            is built from values that genuinely appear in the table.
  selfeval  ask the model "can this be answered from the table?" and treat NO as the
            abstention signal.
  premise   a deterministic structural check: does the question ask for a column that
            does not exist, name an entity that does not exist, or assert a count the
            sheet contradicts? No model at all.
  tokens    completion length, a free degenerate baseline (longer = more invention?)

Reported as AUROC against "the grader said the answer was fabricated". AUROC is used
because the signals are on different scales; the threshold each would use in practice
is a separate question and is printed as a suggestion, not a claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re

import pandas as pd

from adversarial_set import load_table

SCHEMA_TOPICS = {
    # topic -> (question cues, columns that would have to exist to answer it)
    "sales_ranking": (r"best[- ]sell|most sold|top\s*\d|rank(?:ed|ing)?|most popular"
                      r"|sales (?:volume|growth)|units sold|sold last month",
                      "sales"),
    "restock": (r"restock|lead time|when (?:will|did)|last arriv|how many days until"
                r"|average lead time",
                "restock"),
    "supplier": (r"supplier|vendor|carrier|delivery compan", "supplier"),
}


def schema_columns(df: pd.DataFrame) -> set[str]:
    return {str(c).lower() for c in df.columns}


def premise_flags(question: str, df: pd.DataFrame) -> tuple[int, list[str]]:
    """Deterministic structural check. Returns (flagged, reasons).

    Deliberately generic: every rule is about the SCHEMA (does the column exist?) or
    the ENTITY LIST (does the product exist?), never about a particular question.
    """
    q = question.lower()
    cols = schema_columns(df)
    reasons: list[str] = []

    for topic, (cue, needed) in SCHEMA_TOPICS.items():
        if re.search(cue, q) and not any(needed in c for c in cols):
            reasons.append(f"{topic}: asked for data the schema has no column for")

    # Nonexistent variant: a product-like name that is close to a real one but absent.
    names = [str(n) for n in df["Product Name"].tolist()]
    real = {n.lower() for n in names}
    for cand in re.findall(r"[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+){1,7}", question):
        low = cand.lower()
        if low in real:
            continue
        words = set(re.findall(r"[a-z]{3,}", low))
        for rn in real:
            if len(words & set(re.findall(r"[a-z]{3,}", rn))) >= 2:
                reasons.append(f"entity: '{cand}' resembles '{rn}' but is not in the sheet")
                break
        else:
            continue
        break

    # A count asserted in the question that exceeds what the sheet holds.
    word_num = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
    m = re.search(r"\b(" + "|".join(word_num) + r"|\d{1,2})\b[^.]{0,40}?"
                  r"\b(colours|colors|sizes|categories|suppliers|warehouses|items)\b", q)
    if m:
        claimed = word_num.get(m.group(1)) or int(m.group(1))
        cls = m.group(2)
        if cls in ("suppliers", "warehouses"):
            actual = 0
        elif cls in ("colours", "colors"):
            actual = 4
        elif cls == "categories":
            actual = int(df["Category"].nunique())
        elif cls == "sizes":
            actual = 3
        else:
            actual = int(df["Product Name"].nunique())
        if claimed > actual:
            reasons.append(f"count: asked for {claimed} {cls}, sheet has {actual}")

    return int(bool(reasons)), reasons


def relevant_premise(question: str, table: pd.DataFrame, full_ctx: str) -> str:
    """Header plus the one row the question is about.

    HHEM-2.1 has a 512-token window and the whole table is ~760 tokens, so passing it
    all silently truncates the table and scores the answer against half of it. A
    support check only needs the row in question, which also matches how the model
    was trained: short premise, short hypothesis.
    """
    q = question.lower()
    rows = ["| " + " | ".join(str(c) for c in table.columns) + " |"]
    hits = [r for r in table.itertuples(index=False)
            if str(r[1]).lower() in q or str(r[0]).lower() in q]
    if not hits:
        return full_ctx[:1800]  # no row named: fall back to the table head
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in hits[:1]]
    return "\n".join(rows)


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheap signals vs the fabrication labels")
    ap.add_argument("--rows", default="probe_runs/adv_plain.jsonl")
    ap.add_argument("--backend", default="llamacpp", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-selfeval", action="store_true")
    ap.add_argument("--out", default="probe_runs/adv_baselines.jsonl")
    ap.add_argument("--emit-probe",
                    help="write (question, context, label) rows for mute_hidden.py")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.rows, encoding="utf-8") if line.strip()]
    df = load_table()
    usable = [r for r in rows if r["verdict"] != "degenerate"]
    print(f"rows={len(rows)} usable={len(usable)} "
          f"(degenerate excluded: {len(rows) - len(usable)})")

    table = pd.read_excel("d:\\codes\\mute\\excel data\\orders.xlsx", sheet_name="Sheet1")
    # Imported rather than rebuilt: the baseline signals must see exactly the table the
    # model saw, and a duplicated builder already drifted once (six columns vs fourteen).
    from mute_adversarial import table_context
    ctx = table_context(table)

    supporter = None
    try:
        from grounding import Supporter
        supporter = Supporter()
    except Exception as e:  # HHEM needs transformers + the 0.1B weights
        print(f"[warn] HHEM unavailable: {type(e).__name__}: {e}")

    from llm_client import generate

    scored: list[dict] = []
    for i, r in enumerate(usable, 1):
        ans = r["answer"]
        rec = {"id": r["id"], "kind": r["kind"], "verdict": r["verdict"],
               "fabricated": r["verdict"] == "fabricated",
               # Two labelings. The strict one counts only invented content; the
               # unified one also counts a wrong lookup, because a gate cannot tell
               # "read the wrong column" from "made it up" and has to block both.
               "unsupported": r["verdict"] in ("fabricated", "wrong")}
        rec["tokens"] = r.get("completion_tokens") or len(ans.split())
        flag, why = premise_flags(r["question"], df)
        rec["premise"] = flag
        rec["premise_why"] = why
        if supporter is not None:
            try:
                premise = relevant_premise(r["question"], table, ctx)
                rec["hhem"] = float(supporter.score(premise, [f"{r['question']} Answer: {ans}"])[0])
                rec["hhem_premise_chars"] = len(premise)
            except Exception as e:
                rec["hhem"] = float("nan")
                print(f"   [hhem error] {type(e).__name__}: {e}")
        if not args.skip_selfeval:
            meta: dict = {}
            raw = generate(
                "Information:\n{ctx}\n\nQuestion: {q}\n\nCan the question be answered "
                "using ONLY the information above? Reply with exactly one word: YES or NO.".format(
                    ctx=ctx, q=r["question"]),
                backend=args.backend, model=args.model, max_tokens=800, meta=meta)
            rec["selfeval_no"] = int("NO" in raw.upper()[:200])
        scored.append(rec)
        print(f"[{i}/{len(usable)}] {r['kind']:<10} {r['verdict']:<10} "
              f"premise={rec['premise']} hhem={rec.get('hhem', float('nan')):.2f}",
              flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in scored:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.emit_probe:
        # The probe consumes (question, context, label). Label = unsupported, so a
        # wrong lookup and an invented value are the same positive class: a gate
        # cannot tell them apart and has to block both.
        by_id = {r["id"]: r for r in rows}
        with open(args.emit_probe, "w", encoding="utf-8") as f:
            for rec in scored:
                src = by_id[rec["id"]]
                f.write(json.dumps({"id": rec["id"], "kind": rec["kind"],
                                    "question": src["question"], "context": ctx,
                                    "label": int(rec["unsupported"])},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {args.emit_probe}: {len(scored)} rows for mute_hidden.py")

    labels_fab = [rec["fabricated"] for rec in scored]
    labels_uns = [rec["unsupported"] for rec in scored]
    print(f"\nfabricated: {sum(labels_fab)}/{len(labels_fab)}   "
          f"unsupported: {sum(labels_uns)}/{len(labels_uns)}")
    print(f"{'signal':<12}{'AUROC(fab)':>12}{'AUROC(uns)':>12}")
    for key in ("premise", "hhem", "selfeval_no", "tokens"):
        vals = [rec.get(key) for rec in scored]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            continue
        f = [float(v) for v in vals]
        print(f"{key:<12}{auroc(f, labels_fab):>12.3f}{auroc(f, labels_uns):>12.3f}")
    print("\nNOTE: a signal at 0.5 carries no information about fabrication on this set.")

    # Precision/recall for the structural checker, because AUROC alone hides that it
    # may be flagging almost everything.
    tp = sum(1 for r in scored if r["premise"] and r["unsupported"])
    fp = sum(1 for r in scored if r["premise"] and not r["unsupported"])
    fn = sum(1 for r in scored if not r["premise"] and r["unsupported"])
    print(f"premise checker: flagged {tp + fp}/{len(scored)}  "
          f"precision {tp / max(1, tp + fp):.2f}  recall {tp / max(1, tp + fn):.2f}")
    print(f"hhem direction: use 1-hhem; AUROC becomes {1 - auroc([float(r['hhem']) for r in scored], labels_uns):.3f} (unsupported)")


if __name__ == "__main__":
    main()
