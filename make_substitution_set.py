"""Build a substitution set by construction, not by retrieval.

The Laurea corpus cannot carry a probe experiment: its failures are mostly upstream
(truncation, retrieval misses), and a real "the value is present but attached to the
wrong entity" case turns up about three times in the whole collection. So the
failure is built instead of harvested.

An item is a small table plus a question. Two earlier designs were rejected:

  1. Harvest from the document collection. Its failures are mostly upstream
     (truncation, retrieval misses), so most remaining errors contain no model
     decision at all - nothing for a hidden-state probe to detect.
  2. Build tables from the collection's public CSVs. The questions turned into
     geography: "which country is Helsinki in". The model already KNOWS that from
     pretraining, so a correct answer says nothing about whether it read the table,
     and a refusal says nothing about whether it suppressed fabrication. Measured:
     3 of 74 bait items were answered correctly from memory
     (Helsinki->Finland, Gothenburg->Sweden, Seattle->United States).

This design fixes the fatal one. Every value comes from a private grade sheet (one
individual's exam scores and group codes) and every entity is a nonce code, so there
is nothing in pretraining to recall. The table window is constructed; the values keep
their real format.

    | Student | Exam |
    | S007    | 86   |
    | S012    | 41   |
    Question: What exam score did student S007 get?

  label 1 (substitution bait)  the asked row is REMOVED, the other rows stay. All the
                               values are real and same-format, so a text-level
                               checker sees support and passes a wrong number.
  label 0 (answerable control) the asked row is PRESENT. Blocking it is a false alarm.

Both classes come from the same rows, so the only difference is whether the answer is
in the table. Two automated validity gates:

  * no distractor may carry the asked value, or a lucky guess mimics a substitution
  * --leak-check asks every question with NO context; anything the model can answer
    from memory is dropped. This is the gate the geography set failed.

Privacy: entity codes are nonce and their assignment to values is shuffled, so a row
cannot be traced back to a person and no name enters the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

# A value space with too few distinct entries cannot support the question: with two
# outcomes (Passed/Failed) the model's 62.5% prior guess is indistinguishable from a
# substitution, so those columns are excluded mechanically rather than by taste.
MIN_DISTINCT = 8

# One entry per (entity, value) column available in a private table. `pattern` is the
# value's format: it filters out header echoes and blanks, and it is what makes "any
# answer in this shape" checkable later.
PRIVATE_RECIPES = [
    {
        "name": "exam_score",
        "source": "Excel2413157.xlsx",
        "value_col": "Exam",
        "pattern": r"^\d{1,3}$",
        "question": "What exam score did student {entity} get?",
    },
    {
        "name": "group_code",
        "source": "Excel2413157.xlsx",
        "value_col": "Group",
        "pattern": r"^[A-E]\d{2}$",
        "question": "Which group is student {entity} assigned to?",
    },
]

# Two elicitation conditions, because they measure different things:
#   grounded  abstention is allowed, so this measures whether the model WANTS to
#             guess. Expect almost no substitutions here.
#   forced    abstention is forbidden, which is the condition under which the earlier
#             structured-extraction probe got 39% fabrication. Substitution lives here.
PROMPTS = {
    "grounded": (
        "Answer the question using ONLY the table below. If the table does not contain "
        'the answer, reply exactly "not stated".\n\nTable:\n{ctx}\n\nQuestion: {q}\n\nAnswer:'
    ),
    "forced": (
        "Answer the question about the table below. Give the value from the table. "
        "Never say the information is missing, never refuse, never leave it blank - "
        "always give your best answer.\n\nTable:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
    ),
    # Strongest elicitor: a schema with a required key leaves no room to decline.
    # This is the condition under which structured extraction reached 39% fabrication.
    "required_json": (
        "You extract data. Reply with a single JSON object and nothing else.\n"
        'Schema: {{"value": string}} - the key "value" is required and must be a string.\n\n'
        "Table:\n{ctx}\n\nQuestion: {q}\n\nJSON:"
    ),
}

# Models decline in prose far more often than with the word "null": "isn't listed in
# the provided table" is an abstention, and reading it as a normal answer turns the
# abstention rate into a phantom success rate. Kept local so the shared HEDGE_RE in
# mute_structured stays stable for the runs already recorded against it.
_ABSENT_RE = re.compile(
    r"(?:isn't|aren't|wasn't|weren't|is not|are not|not)\b[^.]{0,40}"
    r"\b(?:listed|provided|given|stated|available|present|mentioned|found|specified|in the table|in the provided table)\b"
    r"|(?:doesn't|don't|didn't|does not|do not|did not)\b[^.]{0,20}\b(?:contain|include|list|provide|mention|specify)\b"
    r"|\bno\s+(?:such|entry|row|information|data)\b",
    re.I,
)


def iter_column_values(markdown: str, col_name: str):
    """Yield the cells of `col_name` for every row under any header naming it.

    A sheet often stacks several sub-tables - a student-number block, then the real
    grade table with its own header. Scanning for each header row that names the
    column handles that; assuming the first row is the header finds nothing.

    Docling keeps cells in their own columns, which is the only reason this works:
    the transcript PDFs come out as one merged cell per row, with no row structure
    at all.
    """
    header: list[str] = []
    idx = -1
    for line in markdown.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and all(c and set(c) <= set(":- ") for c in cells):
            continue  # | --- | :--- | separator row
        if col_name in cells:
            header, idx = cells, cells.index(col_name)
            continue
        if idx >= 0 and len(cells) == len(header):
            yield cells[idx]


def load_private_values(corpus: str, source: str, value_col: str,
                        pattern: str) -> list[str]:
    """Every value of one column of one file, in order. Entity identity is dropped."""
    from rag_pro.qwen_rag import _cache_path, collect_files, parse_all

    rx = re.compile(pattern)
    values: list[str] = []
    for rec in parse_all(collect_files(corpus), _cache_path(corpus)):
        if source.lower() not in rec["url"].lower():
            continue
        for page in rec["raw"]:
            values += [v for v in iter_column_values(page["content"], value_col)
                       if rx.match(v)]
    return values


def render_table(rows: list[tuple[str, str]], entity_col: str, value_col: str) -> str:
    lines = [f"| {entity_col} | {value_col} |", "| --- | --- |"]
    lines += [f"| {e} | {v} |" for e, v in rows]
    return "\n".join(lines)


def build(items: list[dict], values: list[str], recipe: dict, rng: random.Random,
          rows: int, limit: int | None) -> None:
    """Turn a list of private values into bait/control item pairs."""
    entity_col, value_col = "Student", recipe["value_col"]
    pairs = [(f"S{i:03d}", v) for i, v in enumerate(values, 1)]
    # Shuffling breaks any residual row-order linkage back to the original sheet.
    rng.shuffle(pairs)
    usable = pairs if not limit else pairs[:limit]
    for entity, value in usable:
        # Two exclusions, both load-bearing:
        #   e != entity  the asked row must not be in the table
        #   v != value   the asked VALUE must not be in the table either, otherwise a
        #                lucky guess would look exactly like a substitution
        others = [(e, v) for e, v in pairs if e != entity and v != value]
        if len(others) < rows - 1:
            continue
        distractors = rng.sample(others, rows - 1)
        full = distractors + [(entity, value)]
        rng.shuffle(full)
        question = recipe["question"].format(entity=entity)
        items.append({
            "kind": "substitution", "label": 1, "recipe": recipe["name"],
            "question": question,
            "context": render_table(distractors, entity_col, value_col),
            "entity": entity, "gold_value": None, "absent_value": value,
            "present_values": sorted({v for _, v in distractors}),
        })
        items.append({
            "kind": "answerable", "label": 0, "recipe": recipe["name"],
            "question": question,
            "context": render_table(full, entity_col, value_col),
            "entity": entity, "gold_value": value, "absent_value": None,
            "present_values": sorted({v for _, v in full}),
        })


def leak_check(items: list[dict], backend: str, model: str | None,
               max_tokens: int) -> list[dict]:
    """Drop questions the model can answer with no context at all.

    The geography set failed this: "which country is Helsinki in" is answerable from
    pretraining, so it measured recall, not grounding. Any dataset built on public
    facts has to pass here before a substitution rate means anything.
    """
    from llm_client import generate
    from mute_structured import final_answer

    gold: dict[str, str] = {}
    for it in items:
        if it["label"] == 0 and it["gold_value"]:
            gold[it["question"]] = it["gold_value"]
    prompt = ('Answer the question. If you do not know, reply exactly "not stated".'
              "\n\nQuestion: {q}\n\nAnswer:")
    leaky: set[str] = set()
    for i, (q, g) in enumerate(gold.items(), 1):
        raw = generate(prompt.format(q=q), backend=backend, model=model,
                       temperature=0.0, max_tokens=max_tokens)
        if g.lower() in (final_answer(raw) or "").lower():
            leaky.add(q)
            print(f"[leak] {q}", flush=True)
        if i % 20 == 0:
            print(f"  leak-checked {i}/{len(gold)}", flush=True)
    print(f"leak check: {len(leaky)}/{len(gold)} questions answerable without context")
    return [it for it in items if it["question"] not in leaky]


def classify(it: dict) -> None:
    """Derive abstained / answered_value / substituted from the stored answer.

    Separate from the model call so a stored run can be re-graded without paying for
    the calls again.
    """
    from mute_structured import is_nullish

    ans = it.get("answer") or ""
    low = ans.lower()
    it["abstained"] = is_nullish(ans) or bool(_ABSENT_RE.search(ans))
    hit = [v for v in it["present_values"] if v and v.lower() in low]
    it["answered_value"] = hit[0] if hit else None
    it["substituted"] = (not it["abstained"]) and bool(hit) and (
        it["gold_value"] is None or hit[0] != it["gold_value"])
    # Not abstaining, not carrying a table value: the answer came from nowhere the
    # table can account for. Usually parametric knowledge, which may even be true.
    it["ungrounded"] = (not it["abstained"]) and not hit


def answer_all(items: list[dict], backend: str, model: str | None, temperature: float,
               condition: str, max_tokens: int) -> None:
    """Ask the model, then classify the answer."""
    from llm_client import generate
    from mute_structured import final_answer, parse_json_obj

    template = PROMPTS[condition]
    for i, it in enumerate(items, 1):
        prompt = template.format(ctx=it["context"], q=it["question"])
        raw = generate(prompt, backend=backend, model=model, temperature=temperature,
                       max_tokens=max_tokens)
        ans = final_answer(raw)
        if condition == "required_json":
            obj = parse_json_obj(ans)
            ans = str(obj.get("value", ans)) if obj else ans
        it["condition"] = condition
        it["answer"] = ans
        classify(it)
        flag = ("SUBSTITUTED" if it["substituted"]
                else "abstained" if it["abstained"]
                else "ungrounded")
        print(f"[{i}/{len(items)}] {it['kind']:<12} {flag:<12} {it['question'][:56]}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Construct a substitution set")
    ap.add_argument("--dir", help="corpus directory holding the CSVs (required to build)")
    ap.add_argument("--out", default="probe_runs/substitution_private.jsonl")
    ap.add_argument("--rows", type=int, default=6, help="rows per constructed table")
    ap.add_argument("--limit", type=int, default=None, help="entities per recipe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak-check", action="store_true",
                    help="drop items answerable without any context")
    ap.add_argument("--answer", action="store_true", help="also query the model")
    ap.add_argument("--reclassify", help="re-grade a stored run without calling the model")
    ap.add_argument("--condition", default="grounded", choices=list(PROMPTS),
                    help="grounded allows abstention; forced forbids it; required_json is strongest")
    ap.add_argument("--backend", default="llamacpp", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    # A reasoning model spends tokens thinking before it answers; at 200 the answer
    # is truncated (or missing entirely) and every metric becomes noise.
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    if args.reclassify:
        with open(args.reclassify, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            classify(r)
        summarize(rows, rows[0].get("condition", "?"))
        with open(args.reclassify, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"re-graded {args.reclassify} in place")
        return

    rng = random.Random(args.seed)
    if not args.dir:
        raise SystemExit("--dir is required to build the set")
    items: list[dict] = []
    for recipe in PRIVATE_RECIPES:
        values = load_private_values(args.dir, recipe["source"], recipe["value_col"],
                                     recipe["pattern"])
        if not values:
            print(f"[skip] {recipe['name']}: no values found in {recipe['source']}")
            continue
        distinct = len(set(values))
        if distinct < MIN_DISTINCT:
            print(f"[skip] {recipe['name']}: only {distinct} distinct values "
                  f"(need {MIN_DISTINCT}); a wrong answer would be a plausible guess")
            continue
        before = len(items)
        build(items, values, recipe, rng, args.rows, args.limit)
        print(f"{recipe['name']:<16} values={len(values):<4} distinct={distinct:<4} "
              f"items=+{len(items) - before}")

    if not items:
        raise SystemExit("no items built: check PRIVATE_RECIPES against the parsed tables")

    if args.leak_check:
        items = leak_check(items, args.backend, args.model, args.max_tokens)
        if not items:
            raise SystemExit("every item leaked: the source is public knowledge")

    if args.answer:
        answer_all(items, args.backend, args.model, args.temperature, args.condition,
                   args.max_tokens)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n = len(items)
    pos = sum(1 for d in items if d["label"] == 1)
    print(f"\nwrote {args.out}: n={n}  substitution={pos}  answerable={n - pos}")
    if args.answer:
        summarize(items, args.condition)


def summarize(items: list[dict], condition: str) -> None:
    bait = [d for d in items if d["label"] == 1]
    ctrl = [d for d in items if d["label"] == 0]
    sub = sum(1 for d in bait if d.get("substituted"))
    abst = sum(1 for d in bait if d.get("abstained"))
    ung = sum(1 for d in bait if d.get("ungrounded"))
    ctrl_ok = sum(1 for d in ctrl if (d.get("answered_value") or "") == d["gold_value"])
    print(f"condition={condition}")
    print(f"bait    : substituted {sub}/{len(bait)}, abstained {abst}/{len(bait)}, "
          f"ungrounded {ung}/{len(bait)}")
    print(f"control : answered correctly {ctrl_ok}/{len(ctrl)}")
    if sub == 0:
        print("NOTE: zero substitutions. If abstention was allowed, that is the "
              "expected result - re-run with --condition forced or required_json.")


if __name__ == "__main__":
    main()
