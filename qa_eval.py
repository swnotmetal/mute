"""Load the hand-annotated question file into (question, gold) pairs.

Format: a question line, optionally followed by a line starting with '-'
holding the expected answer. Blank lines separate entries.
"""

from __future__ import annotations

import argparse
import pathlib
import re


def derive_keys(gold: str) -> list[str]:
    """Key terms an answer must contain. Numbers, quoted text and capitalised
    words come first; short answers fall back to the whole string."""
    keys: list[str] = []
    keys += re.findall(r"\b\d{1,4}(?:\.\d+)?\b", gold)
    keys += [q.strip() for q in re.findall(r'["\u201c]([^"\u201d]{3,40})["\u201d]', gold)]
    keys += [w for w in re.findall(r"\b[A-Z][A-Za-z0-9_.+-]{2,}\b", gold)]
    seen, out = set(), []
    for k in keys:
        low = k.lower()
        if low not in seen:
            seen.add(low)
            out.append(k)
    return out[:6] or [gold.strip()[:40]]


def load_qa(path: str) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-"):
            answer = line.lstrip("-").strip()
            if pairs and pairs[-1][1] is None:
                pairs[-1] = (pairs[-1][0], answer)
            else:
                pairs.append((answer, None))  # stray answer line, keep it visible
        else:
            pairs.append((line, None))
    return pairs


# Hand-written grading rules, keyed by a unique prefix of the question.
# groups   : every group must match; a group matches if ANY of its terms appears
# any_of   : at least `min_hits` of these terms must appear
# kind     : fact | count | negative | list | unclear (unclear is excluded)
RULES: list[tuple[str, dict]] = [
    ("Who was the contact person", {"kind": "fact", "groups": [["daria"], ["galina"]]}),
    ("What does the BOP_python.pdf cover", {"kind": "fact",
                                           "groups": [["python"], ["basic", "programming", "basics"]]}),
    ("What does the Employment Certificate", {"kind": "fact",
                                              "groups": [["bravori"], ["herizon"]]}),
    ("What did the student build in the Ionic project", {"kind": "negative",
                                                         "groups": [["niilo"],
                                                                    ["review", "not", "did not", "didn't", "no "]]}),
    ("How many files relate to mobile app", {"kind": "count", "groups": [["9"]]}),
    ("What does the \"elements of AI\" certificate", {"kind": "fact",
                                                      "groups": [["mooc"], ["ai", "artificial"]]}),
    ("Who was the student's partner in the Simple Car Rental", {"kind": "fact", "groups": [["samu"]]}),
    ("What does the JavaScript report", {"kind": "fact",
                                        "groups": [["javascript"], ["to-do", "todo", "to do"]]}),
    ("What Linux skills are recorded", {"kind": "list",
                                       "any_of": ["sudo", "awk", "bash", "command", "calculator",
                                                  "bubble"], "min_hits": 2}),
    ("Which phishing-related files exist", {"kind": "list",
                                            "any_of": ["phishing-email-practise", "phishing box",
                                                       "phishing-iq", "phishing iq"], "min_hits": 2}),
    ("How many Power BI (.pbix) files", {"kind": "count", "groups": [["2", "two"]]}),
    ("What does the OS assignment cover", {"kind": "fact", "groups": [["windows"]]}),
    ("What is the \"Skylark\" R&D plan", {"kind": "fact",
                                         "groups": [["nokia"], ["validity"]]}),
    ("What was the student's PicoCTF reporting", {"kind": "list",
                                                  "any_of": ["obedient cat", "python wrangling",
                                                             "net cat", "nice netcat", "super ssh",
                                                             "bookmarklet"], "min_hits": 3}),
    ("What did the student learn from the phishing IQ test? (rephrase: how would you explain",
     {"kind": "unclear"}),
    ("What business process did the student model", {"kind": "fact",
                                                     "groups": [["customer"], ["order"]]}),
    ("What was the placement contract's weekly hour requirement",
     {"kind": "negative", "any_of": ["400", "not specified", "did not", "no weekly"], "min_hits": 1}),
    ("What was the student's student number", {"kind": "fact", "groups": [["2413157"]]}),
    ("What did the student write in the placement contract about confidentiality",
     {"kind": "negative",
      "groups": [["confidential"], ["not", "no ", "did not", "didn't", "absent"]]}),
    ("What is the difference between the two signed PDFs", {"kind": "unclear"}),
]


def _has(text: str, term: str) -> bool:
    """Substring match, with word boundaries for short or numeric terms."""
    if term[0].isdigit() or len(term) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def grade(answer: str, rule: dict) -> bool:
    t = answer.lower()
    if "groups" in rule:
        return all(any(_has(t, term) for term in grp) for grp in rule["groups"])
    hits = sum(1 for term in rule.get("any_of", []) if _has(t, term))
    return hits >= rule.get("min_hits", 1)


def rule_for(question: str) -> dict | None:
    for prefix, rule in RULES:
        if question.startswith(prefix):
            return rule
    return None


def run(questions: list[tuple[str, str]], args) -> None:
    """Query the RAG for each question and append results to JSONL."""
    import json
    import os

    from llm_client import generate
    from mute_structured import build_context, final_answer
    from rag_pro.eval_rag import REFUSAL_MARKERS
    from rag_pro.qwen_rag import PROMPTS, _cache_path, collect_files, parse_all, retrieve_context
    from rag_pro.rerank import Reranker

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    continue
    todo = [p for p in questions if p[0] not in done]
    print(f"total={len(questions)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for i, (q, gold) in enumerate(todo, 1):
            _q, kept, scores, _n = retrieve_context(
                records, q, retrieve_k=20, top_k=5, reranker=reranker, use_keywords=False)
            ctx = build_context(kept)
            raw = generate(f"{PROMPTS['grounded']}\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:",
                           backend=args.backend, model=args.model, max_tokens=800)
            answer = final_answer(raw)
            rec = {"question": q, "gold": gold, "answer": answer,
                   "refused": any(m in answer.lower() for m in REFUSAL_MARKERS),
                   "max_rr": max(scores) if scores else 0.0, "n_kept": len(kept)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {q[:60]}", flush=True)


def report(path: str, verbose: bool = False) -> None:
    import json
    from collections import defaultdict

    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    refusals = 0
    rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rule = rule_for(r["question"])
            if not rule or rule["kind"] == "unclear":
                continue
            rows += 1
            ok = grade(r["answer"], rule)
            by_kind[rule["kind"]][0] += int(ok)
            by_kind[rule["kind"]][1] += 1
            refusals += int(r["refused"])
            if verbose:
                ans = " ".join(r["answer"].split())[:150]
                print(f"[{'PASS' if ok else 'FAIL'}] {rule['kind']:<9} {r['question'][:56]}")
                print(f"        gold: {' '.join(r['gold'].split())[:110]}")
                print(f"        ans : {ans}\n")

    print(f"{'kind':<10}{'pass':>6}{'total':>7}{'rate':>8}")
    for kind, (ok, n) in sorted(by_kind.items()):
        print(f"{kind:<10}{ok:>6}{n:>7}{ok / n:>8.0%}")
    total_ok = sum(v[0] for v in by_kind.values())
    print(f"{'ALL':<10}{total_ok:>6}{rows:>7}{total_ok / rows:>8.0%}")
    print(f"refusals: {refusals}/{rows}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="rag_pro/test_questions.txt")
    ap.add_argument("--keys", action="store_true", help="show auto-derived keys (debug)")
    ap.add_argument("--run", action="store_true", help="query the RAG and save answers")
    ap.add_argument("--report", action="store_true", help="grade saved answers")
    ap.add_argument("-v", "--verbose", action="store_true", help="show each graded row")
    ap.add_argument("--dir", help="corpus directory (required with --run)")
    ap.add_argument("--backend", default="llamacpp", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="probe_runs/qa.jsonl")
    args = ap.parse_args()

    pairs = load_qa(args.questions)
    labelled = [(q, g) for q, g in pairs if g]

    if args.report:
        report(args.out, args.verbose)
        return
    if args.run:
        run(labelled, args)
        return

    print(f"total entries: {len(pairs)}   labelled: {len(labelled)}   "
          f"unlabelled: {len(pairs) - len(labelled)}\n")
    for i, (q, g) in enumerate(labelled, 1):
        rule = rule_for(q)
        kind = rule["kind"] if rule else "NO RULE"
        desc = (rule.get("groups") or f"any{rule.get('min_hits', 1)} of {rule.get('any_of')}") \
            if rule else "-"
        print(f"{i:2d}. [{kind}] {q[:80]}")
        print(f"    gold: {g[:100]}")
        print(f"    rule: {desc}")
        if args.keys:
            print(f"    auto: {derive_keys(g)}")


if __name__ == "__main__":
    main()
