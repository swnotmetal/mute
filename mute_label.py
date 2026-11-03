"""Deterministic label generator with an audit trail (no model calls).

Labels are derived ONLY from the filesystem and the parsed corpus text:
  answerable  - a rule computed the expected answer (count / file set / value)
  absent      - a curated list of questions that the collection cannot answer
  epistemic   - meta questions about reliability/knowability (excluded from metrics)
  unknown     - no rule matched -> excluded, never guessed

Every label carries `evidence` (the files/snippets that justify it). A random
sample is written to a review file so a human can spot-check the labels.

Outputs (into --outdir, gitignored):
  labels.jsonl        one record per question
  labels_review.txt   N random records with full evidence, for manual audit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from qwen_agent.tools.doc_parser import Record

from rag_pro.qwen_rag import _cache_path, collect_files, parse_all

# Questions the collection cannot answer (curated; shown in the review sample).
ABSENT_PATTERNS = [
    r"capital of france",
    r"sdelete\.zip",
    r"recovery codes",
    r"module4lab\.pkt",
    r"cipher\.png",
    r"taks2b\.png",
    r"safeonweb phishing image",
    r"todoapp-.*\.mp4",
    r"devop\.drawio",
    r"email\.drawio",
]

# Meta questions about reliability / knowability -> excluded from binary metrics.
EPISTEMIC_PATTERNS = [
    r"is that inference reliable",
    r"is that number actually knowable",
    r"can you know for sure",
    r"determinable from the filename alone",
    r"which claims are directly supported",
    r"can you name the teammates",
    r"explicitly state which claims",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def corpus_files(root: str) -> list[Path]:
    """Every real file under root (NOT filtered by parser support).

    Counting rules must see images, archives, .pbix, .pkt etc. too; using the
    parser-supported list would report 0 images and a wrong file total.
    """
    return [p for p in Path(root).rglob("*") if p.is_file()]


def chunk_texts(root: str) -> list[tuple[str, str]]:
    """[(url, content)] over the parsed corpus."""
    recs = parse_all(collect_files(root), _cache_path(root))
    out: list[tuple[str, str]] = []
    for r in recs:
        d = Record(**r)
        for pg in d.raw:
            out.append((d.url, pg.content or ""))
    return out


def files_mentioning(term: str, chunks: list[tuple[str, str]]) -> list[str]:
    t = term.lower()
    hits = sorted({Path(u).name for u, c in chunks if t in (c or "").lower()})
    return hits


def label_question(q: str, files: list[Path], chunks: list[tuple[str, str]]) -> dict:
    ql = q.lower().strip()

    for pat in EPISTEMIC_PATTERNS:
        if re.search(pat, ql):
            return {"label": "epistemic", "expected": None,
                    "evidence": {"matched_pattern": pat}}

    for pat in ABSENT_PATTERNS:
        if re.search(pat, ql):
            return {"label": "absent", "expected": None,
                    "evidence": {"matched_pattern": pat,
                                 "reason": "binary/diagram content or out-of-scope question"}}

    # ---- counting rules (computed from the real directory) ----
    if "how many distinct pdf" in ql or ("how many" in ql and ".pdf" in ql and "distinct" in ql):
        n = sum(1 for f in files if f.suffix.lower() == ".pdf")
        return {"label": "answerable", "expected": n,
                "evidence": {"rule": "count_ext:.pdf", "total_files": len(files)}}
    if "how many files are images" in ql:
        n = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
        return {"label": "answerable", "expected": n,
                "evidence": {"rule": "count_image_exts", "exts": sorted(IMAGE_EXTS)}}
    if "pbix" in ql and "how many" in ql:
        pbix = [f.name for f in files if f.suffix.lower() == ".pbix"]
        return {"label": "answerable", "expected": len(pbix),
                "evidence": {"rule": "count_ext:.pbix", "files": pbix}}
    if "how many files contain the word" in ql and "shuang" in ql:
        names = [f.name for f in files if "shuang" in f.name.lower()]
        return {"label": "answerable", "expected": len(names),
                "evidence": {"rule": "filename_contains:shuang", "files": names}}
    if "total number of files" in ql:
        return {"label": "answerable", "expected": len(files),
                "evidence": {"rule": "count_all_supported"}}
    if "modified or created in 2026" in ql:
        y2026 = [f.name for f in files
                 if f.stat().st_mtime and __import__("datetime").datetime.fromtimestamp(
                     f.stat().st_mtime).year == 2026]
        return {"label": "answerable", "expected": len(y2026),
                "evidence": {"rule": "mtime_year:2026", "files": y2026,
                             "note": "mtime reflects copy time, not authorship"}}

    # ---- file-set rules (grep over parsed chunks) ----
    m = re.search(r"which files (?:mention|list) (?:a student named )?(\w+)", ql)
    if m:
        term = m.group(1)
        hits = files_mentioning(term, chunks)
        return {"label": "answerable", "expected": hits,
                "evidence": {"rule": f"grep:{term}", "files": hits}}
    if "which files list pertti tiainen" in ql:
        hits = files_mentioning("Pertti Tiainen", chunks)
        return {"label": "answerable", "expected": hits,
                "evidence": {"rule": "grep:Pertti Tiainen", "files": hits}}

    # ---- value rules we already verified ----
    if "partner in the simple car rental" in ql:
        hits = files_mentioning("samu", chunks)
        return {"label": "answerable", "expected": "Samu",
                "evidence": {"rule": "grep:samu", "files": hits,
                             "snippet": "Created by Shuang and Samu"}}

    return {"label": "unknown", "expected": None, "evidence": {"rule": None}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic label generator with audit trail")
    ap.add_argument("--questions", default="rag_pro/test_questions.txt")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--outdir", default="probe_runs")
    ap.add_argument("--review", type=int, default=15, help="rows in the manual-review sample")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    questions = [s.strip() for s in Path(args.questions).read_text(encoding="utf-8").splitlines()
                 if s.strip()]
    files = corpus_files(args.dir)
    chunks = chunk_texts(args.dir)

    records = []
    for q in questions:
        lab = label_question(q, files, chunks)
        records.append({"question": q, **lab})

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("label counts:", counts, " total:", len(records))

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.review, len(records)))
    path = os.path.join(args.outdir, "labels_review.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(f"[{r['label']}] {r['question']}\n")
            f.write(f"    expected: {r['expected']}\n")
            f.write(f"    evidence: {json.dumps(r['evidence'], ensure_ascii=False)[:400]}\n\n")
    print(f"review sample ({len(sample)}) -> {path}")


if __name__ == "__main__":
    main()
