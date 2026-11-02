"""RAG pass-line eval: a small question set with expected key terms.

Run after any pipeline change. If the pass rate clears PASS_BAR, the local RAG is
"good enough" and the next phase is the Mute probe (retrieval-failure detection).
"""

from __future__ import annotations

import argparse

from .qwen_rag import _cache_path, answer_question, collect_files, parse_all
from .rerank import Reranker

# (question, acceptable answer substrings incl. Chinese, since the 7B may answer in zh).
EVAL = [
    ("What did the student do with a fellow student named Samu?", ["samu", "samo"]),
    ("What is the car rental management system about?", ["rental", "租赁", "租车"]),
    ("What did the student learn about phishing?", ["phish", "钓鱼"]),
    ("Who signed the work placement contract?", ["shuang", "吴", "grén"]),
    ("What did the student learn about POWER-BI?", ["power bi", "dax", "power"]),
    ("What is the student's name on the placement contract?", ["shuang", "吴"]),
    ("What does the GDPR assignment cover?", ["gdpr", "data protection", "隐私"]),
    ("What did the student learn about Linux?", ["linux"]),
]

REFUSAL_MARKERS = [
    "资料中没有相关信息", "no mention", "no information", "not mentioned", "not contain",
    "no explicit mention", "cannot find", "not provided", "doesn't mention", "not specified",
    "unable to find", "does not mention", "there is no", "no relevant", "not discussed",
    "no data", "did not explicitly", "cannot help", "does not provide", "doesn't provide",
    "not directly", "no details", "not include", "does not contain", "no specific",
]
PASS_BAR = 0.75  # 6/8


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG pass-line eval")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--retrieve-k", type=int, default=20, help="K1: hybrid candidates before rerank")
    ap.add_argument("--top-k", type=int, default=5, help="K2: chunks kept for the generator")
    ap.add_argument("--no-rerank", action="store_true", help="skip the multilingual cross-encoder reranker")
    args = ap.parse_args()

    records = parse_all(collect_files(args.dir), _cache_path(args.dir))
    print(f"records: {len(records)}")

    reranker = None if args.no_rerank else Reranker()
    passed = 0
    for q, alts in EVAL:
        ans, _query, _kept = answer_question(records, q, retrieve_k=args.retrieve_k,
                                             top_k=args.top_k, rerank=not args.no_rerank,
                                             reranker=reranker)
        low = ans.lower()
        ok = (not any(m in low for m in REFUSAL_MARKERS)) and any(a in low for a in alts)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        print(f"        -> {ans[:180].replace(chr(10), ' ')}")

    rate = passed / len(EVAL)
    print(f"\n=== RESULT: {passed}/{len(EVAL)} = {rate:.0%} (bar {PASS_BAR:.0%}) ===")
    print(">>> RAG 达到及格线：可以开始写 Mute 探针（检索失败检测）" if rate >= PASS_BAR
          else ">>> 未达及格线：继续调检索/解析")

if __name__ == "__main__":
    main()
