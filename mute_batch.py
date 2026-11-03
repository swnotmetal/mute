"""Batch runner over a question file with JSONL checkpointing.

Per question:
  1. retrieve + rerank   (use_keywords=False -> Ollama only for bge-m3, no LLM)
  2. grounded answer     (+ refusal flag)
  3. LLM self-eval       ("can the context answer this? YES/NO")
  4. optional forced answer with --forced (no escape hatch)

Each result is appended as one JSON line; questions already present in --out are
skipped, so a crash / Ctrl-C can be resumed.

Example:
  python mute_batch.py --questions rag_pro/test_questions.txt --out probe_runs/q7b.jsonl \
      --dir "C:/Users/shuangboi/Desktop/Laurea" --backend llamacpp --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

from llm_client import generate
from mute_structured import build_context, final_answer
from rag_pro.eval_rag import REFUSAL_MARKERS
from rag_pro.qwen_rag import PROMPTS, _cache_path, collect_files, parse_all, retrieve_context
from rag_pro.rerank import Reranker

SELFEVAL = (
    "Information:\n{ctx}\n\nQuestion: {q}\n\n"
    "Can the question be answered using ONLY the information above? "
    "Reply with exactly one word: YES or NO."
)


def load_questions(path: str) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def load_done(path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpointed batch probe runner")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--dir", required=True, help="corpus directory")
    ap.add_argument("--out", default="probe_runs/questions.jsonl")
    ap.add_argument("--backend", default="llamacpp", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="run at most N pending questions")
    ap.add_argument("--forced", action="store_true",
                    help="also generate with the no-escape-hatch prompt")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    done = load_done(args.out)
    todo = [q for q in questions if q not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"total={len(questions)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    reranker = Reranker()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            _query, kept, scores, _n = retrieve_context(
                records, q, retrieve_k=args.retrieve_k, top_k=args.top_k,
                reranker=reranker, use_keywords=False)
            ctx = build_context(kept)
            max_rr = max(scores) if scores else 0.0

            grounded = generate(
                f"{PROMPTS['grounded']}\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:",
                backend=args.backend, model=args.model, max_tokens=800)
            answer = final_answer(grounded)
            refused = any(m in answer.lower() for m in REFUSAL_MARKERS)

            se = generate(SELFEVAL.format(ctx=ctx, q=q),
                          backend=args.backend, model=args.model, max_tokens=300)
            self_no = final_answer(se).strip().upper().startswith("NO")

            rec = {"question": q, "max_rr": max_rr, "n_kept": len(kept),
                   "top_files": [os.path.basename(u) for u, _ in kept[:3]],
                   "answer": answer, "empty": not answer.strip(),
                   "refused": refused, "self_no": self_no,
                   "secs": round(time.time() - t0, 1)}

            if args.forced:
                forced = generate(
                    f"{PROMPTS['helpful']}\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:",
                    backend=args.backend, model=args.model, max_tokens=800)
                rec["forced_answer"] = forced
                rec["forced_refused"] = any(m in forced.lower() for m in REFUSAL_MARKERS)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] rr={max_rr:.4f} n={len(kept)} "
                  f"refused={refused} selfNO={self_no} {rec['secs']}s :: {q[:56]}", flush=True)


if __name__ == "__main__":
    main()
