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
import json
import os

from llm_client import generate
from mute_structured import (
    FIELDS,
    build_context,
    build_prompt,
    final_answer,
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


def load_rows(path: str) -> list[tuple]:
    """Rows already computed in an earlier run (crash-safe resume)."""
    rows: list[tuple] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rows.append((d["key"], d["expected"], d["value"], d["fabricated"],
                             d["correct"], d["max_rr"], d["self_no"], d["ungrounded"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare abstention signals against fabrication GT")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--grounding", default="span", choices=["span", "hhem"],
                    help="span = substring check; hhem = Vectara HHEM-2.1-Open")
    ap.add_argument("--limit", type=int, default=0, help="only the first N fields")
    ap.add_argument("--out", default="probe_runs/compare.jsonl",
                    help="JSONL checkpoint; finished fields are skipped on restart")
    args = ap.parse_args()

    fields = FIELDS[:args.limit] if args.limit else FIELDS
    supporter = None
    if args.grounding == "hhem":
        from grounding import Supporter

        supporter = Supporter()
    print(f"grounding signal: {args.grounding}", flush=True)

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()

    keys = {f[0] for f in fields}
    rows = [r for r in load_rows(args.out) if r[0] in keys]  # drop fields no longer listed
    done = {r[0] for r in rows}
    todo = [f for f in fields if f[0] not in done]
    print(f"fields={len(fields)} done={len(done)} todo={len(todo)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for key, question, expected in todo:
            _q, kept, scores, _n = retrieve_context(
                records, question, retrieve_k=args.retrieve_k, top_k=args.top_k,
                reranker=reranker, use_keywords=False)
            ctx = build_context(kept)
            max_rr = max(scores) if scores else 0.0

            out = generate(build_prompt(question, key, ctx, allow_null=False),
                           backend=args.backend, model=args.model,
                           temperature=args.temperature, max_tokens=800)
            obj = parse_json_obj(out)
            value = None if obj is None else obj.get(key)

            if expected is None:
                fabricated = not is_nullish(value)
                correct = not fabricated
            else:
                fabricated = False
                correct = (not is_nullish(value)) and expected in str(value).lower()

            se = generate(SELFEVAL.format(ctx=ctx, q=question),
                          backend=args.backend, model=args.model,
                          temperature=0.0, max_tokens=300)
            self_no = final_answer(se).strip().upper().startswith("NO")
            if is_nullish(value):
                ungrounded = False
            elif supporter is not None:
                ungrounded = supporter.score(ctx, [f"{question} Answer: {value}"])[0] < 0.5
            else:
                ungrounded = not value_in_context(value, ctx)

            row = (key, expected, value, fabricated, correct, max_rr, self_no, ungrounded)
            rows.append(row)
            fh.write(json.dumps({
                "key": key, "expected": expected, "value": value,
                "fabricated": fabricated, "correct": correct, "max_rr": max_rr,
                "self_no": self_no, "ungrounded": ungrounded,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {key:<22} fab={fabricated} ungrnd={ungrounded} {max_rr:.4f}", flush=True)

    print(f"{'field':<16}{'truth':>8}{'value':>16}{'fab':>6}{'ok':>6}{'max_rr':>8}{'selfNO':>7}{'ungrnd':>7}")
    print("-" * 74)
    for key, expected, value, fab, cor, mr, se_no, ung in rows:
        truth = "ABSENT" if expected is None else expected
        print(f"{key:<16}{truth:>8}{str(value)[:14]:>16}{str(fab):>6}{str(cor):>6}"
              f"{mr:>8.4f}{str(se_no):>7}{str(ung):>7}")

    absent = [r for r in rows if r[1] is None]
    controls = [r for r in rows if r[1] is not None]
    fab_total = sum(1 for r in absent if r[3])
    print(f"\nfabricated on absent fields : {fab_total}/{len(absent)}")
    print(f"controls correct            : {sum(1 for r in controls if r[4])}/{len(controls)}")

    print("\nAs ABSTENTION POLICIES:")
    se_fire = [r for r in rows if r[6]]
    print(f"  B. self-eval NO fires {len(se_fire)}/{len(rows)} | prevents "
          f"{sum(1 for r in se_fire if r[3])}/{fab_total} fabrications | "
          f"false-abstain {sum(1 for r in se_fire if r[1] is not None)}/{len(controls)}")
    sp_fire = [r for r in rows if r[7]]
    print(f"  C. span ungrounded fires {len(sp_fire)}/{len(rows)} | catches "
          f"{sum(1 for r in sp_fire if r[3])}/{fab_total} fabrications | "
          f"false-abstain {sum(1 for r in sp_fire if r[1] is not None)}/{len(controls)}")
    print(f"  A. retrieval: max_rr(fabricated) = {[round(r[5], 4) for r in absent if r[3]]}")
    print(f"                max_rr(abstained)  = {[round(r[5], 4) for r in absent if not r[3]]}")
    print("  D. MUTE hidden-state probe = NOT wired (needs the CUAD probe model)")

    print("\nFabrications by type:")
    fab = [r for r in rows if r[3]]
    base = [r for r in fab if r[7]]
    subs = [r for r in fab if not r[7]]
    print(f"  total {len(fab)}: baseless {len(base)} (text-detectable), "
          f"substitution {len(subs)} (value exists in context -> text-invisible)")
    print(f"  self-eval fired on {sum(1 for r in fab if r[6])}/{len(fab)} fabrications")
    if subs:
        print("  substitution cases:", [r[0] for r in subs])

    print("\nNOTE: pilot. 'ungrnd=False' on a fabricated value = the fake value happens to "
          "appear in context (answer substitution / label audit needed).")


if __name__ == "__main__":
    main()
