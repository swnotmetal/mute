"""Hidden-state extraction for the MUTE probe, with a selectable pooling site.

Why this exists
---------------
The notebook probe reads the LAST prompt token at one layer. That is the right
question for "does the model know?" - the final token has attended over the whole
prompt, so it aggregates everything. It is a questionable place to ask "did the
model attach a real value to the wrong question?", because a *substitution* is
decided locally, at the value's own tokens, long before the last position.

`pool` makes that choice explicit and switchable:

    last      index -1 (what the notebook does)
    question  mean over the question's tokens - "what is being asked"
    digits    mean over prompt tokens containing a digit - where a numeric
              substitution has to be resolved
    all       mean over every prompt token (control; washes out local signals)

No sklearn: this ships a closed-form ridge probe (a linear probe with an L2
penalty), so the comparison runs on numpy alone.

Nothing here changes the RAG. The context is whatever the frozen pipeline returns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SYSTEM = "Extract the exact requested value from the email."
POOLS = ("last", "question", "digits", "all")
# 0 = the context answers the question, 1 = it does not (the interesting class).
# `unknown` rows are unverified and are dropped: a probe trained on them would be
# learning the annotator's uncertainty, not the model's.
LABEL_MAP = {"answerable": 0, "absent": 1, "epistemic": 1}


def _subsequence_at(haystack: list[int], needle: list[int]) -> int:
    """Index of the LAST occurrence of `needle` in `haystack`, or -1."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return -1
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start:start + n] == needle:
            return start
    return -1


class HiddenReader:
    """One forward pass per (context, question); every pool is a view of it.

    The model is loaded lazily and never downloaded implicitly: a missing cache
    raises instead of silently pulling ~1.7 GB.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None,
                 allow_download: bool = False) -> None:
        self.model_id = model_id
        self.device = device
        self.allow_download = allow_download
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=not self.allow_download)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 only pays off on GPU; CPU inference is faster and better supported
        # in fp32, and the probe input is a single vector per sample.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            from transformers import Qwen3_5ForConditionalGeneration as Loader
        except ImportError:
            from transformers import AutoModelForCausalLM as Loader
        try:
            self.model = Loader.from_pretrained(
                self.model_id, dtype=dtype,
                local_files_only=not self.allow_download).to(device).eval()
        except OSError as e:
            raise RuntimeError(
                f"{self.model_id} is not in the local HF cache and downloads are "
                f"disabled. Re-run with --allow-download (about 1.7 GB)."
            ) from e
        self.device = device

    def read(self, context: str, question: str, layer: int = 15,
             pool: str = "last") -> np.ndarray:
        """Return the pooled hidden state at `layer` as a 1-D float32 vector."""
        return self.read_pools(context, question, layer, (pool,))[pool]

    def read_pools(self, context: str, question: str, layer: int = 15,
                   pools: tuple[str, ...] = POOLS) -> dict[str, np.ndarray]:
        """One forward pass, every requested pooling of it.

        A forward pass costs about 20 s on this CPU build, so reading each pooling
        separately multiplied the run by four for no reason.
        """
        grid = self.read_grid(context, question, (layer,), pools)
        return {p: grid[(layer, p)] for p in pools}

    def read_grid(self, context: str, question: str, layers: tuple[int, ...],
                  pools: tuple[str, ...] = POOLS) -> dict[tuple[int, str], np.ndarray]:
        """One forward pass, every (layer, pooling) pair. Same cost as one read.

        The model already computes all 25 hidden states per forward, so a layer sweep
        is free once the input has been run - which is why it is worth doing before
        collecting more data.
        """
        for p in pools:
            if p not in POOLS:
                raise ValueError(f"pool must be one of {POOLS}, got {p!r}")
        self._load()
        import torch

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Email: {context}\nQuestion: {question}"}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        ids = inputs["input_ids"][0].tolist()
        grid: dict[tuple[int, str], np.ndarray] = {}
        for layer in layers:
            # hidden_states[0] is the embedding output, so index == layer number.
            h = out.hidden_states[layer][0].to(torch.float32).cpu().numpy()
            for pool in pools:
                grid[(layer, pool)] = self._pool(h, ids, question, pool)
        return grid

    def _pool(self, h: np.ndarray, ids: list[int], question: str,
              pool: str) -> np.ndarray:
        if pool == "last":
            return h[-1]
        if pool == "all":
            return h.mean(axis=0)
        if pool == "question":
            q_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            start = _subsequence_at(ids, q_ids)
            if start < 0:
                # Fall back to the tail: the question is always the last thing in
                # the prompt, so the final |q| positions are the right span anyway.
                start = max(0, len(ids) - len(q_ids))
            return h[start:start + len(q_ids)].mean(axis=0)
        # digits: tokens whose surface form contains a digit. A substitution has to
        # be resolved here, where the model picks which number to carry forward.
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        idx = [i for i, p in enumerate(pieces) if any(c.isdigit() for c in p)]
        if not idx:
            return h[-1]
        return h[idx].mean(axis=0)


def ridge_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                 l2: float = 1.0) -> np.ndarray:
    """Closed-form L2 linear probe: w = (X'X + lI)^-1 X'y, on standardised features."""
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    xs = (x_train - mu) / sd
    ys = y_train - y_train.mean()
    a = xs.T @ xs + l2 * np.eye(xs.shape[1])
    w = np.linalg.solve(a, xs.T @ ys)
    return ((x_test - mu) / sd) @ w


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cv_auroc(x: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = 1.0,
             seed: int = 0, pca: int | None = None) -> float:
    """Folds are shuffled once; every sample is scored by a probe that never saw it.

    `pca` projects each fold onto its own principal components, fitted on the training
    part only. With n around 40 and d = 1024 a raw linear probe cannot be fitted at
    all, so this is not an optimisation - without it every result is a coin flip.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        if len(np.unique(y[train])) < 2:
            continue
        xt, xv = x[train], x[test]
        if pca:
            mu = xt.mean(axis=0)
            centred = xt - mu
            _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            comp = min(pca, vt.shape[0])
            basis = vt[:comp]
            xt = centred @ basis.T
            xv = (xv - mu) @ basis.T
        scores[test] = ridge_scores(xt, y[train], xv, l2=l2)
    return auroc(scores, y)


def build_dataset(labels_path: str, corpus_dir: str, limit: int | None = None):
    """Retrieve a context per labelled question using the FROZEN RAG pipeline."""
    from mute_structured import build_context
    from rag_pro.qwen_rag import _cache_path, collect_files, parse_all, retrieve_context
    from rag_pro.rerank import Reranker

    rows = []
    skipped = 0
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") not in LABEL_MAP:
                skipped += 1
                continue
            rows.append({"question": r["question"], "label": LABEL_MAP[r["label"]]})
    print(f"usable={len(rows)}  skipped(unverified)={skipped}", flush=True)
    if not rows:
        raise SystemExit("no labelled rows: annotate some labels.jsonl entries first")
    if limit:
        rows = rows[:limit]
    records = parse_all(collect_files(corpus_dir), _cache_path(corpus_dir))
    reranker = Reranker()
    dataset = []
    for i, r in enumerate(rows, 1):
        _q, kept, _s, _n = retrieve_context(records, r["question"], retrieve_k=20,
                                            top_k=5, reranker=reranker, use_keywords=False)
        dataset.append({"question": r["question"], "context": build_context(kept),
                        "label": int(r["label"])})
        print(f"[{i}/{len(rows)}] retrieved for: {r['question'][:60]}", flush=True)
    print(f"label balance: {sum(d['label'] for d in dataset)} missing / "
          f"{len(dataset) - sum(d['label'] for d in dataset)} answerable", flush=True)
    return dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-pooling AUROC of a hidden-state probe")
    ap.add_argument("--dataset", help="JSONL with question/context/label")
    ap.add_argument("--labels", help="labels.jsonl (question/label) to retrieve context for")
    ap.add_argument("--dir", help="corpus directory (with --labels)")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--layers", default="",
                    help="comma separated layers for a sweep; free, one forward each")
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--pca", type=int, default=15,
                    help="components per fold; needed when n is small (0 disables)")
    ap.add_argument("--permutations", type=int, default=20,
                    help="shuffled-label runs, to show what chance looks like at this n")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default="probe_runs/pool_eval.json")
    args = ap.parse_args()

    if args.dataset:
        with open(args.dataset, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    elif args.labels and args.dir:
        dataset = build_dataset(args.labels, args.dir, args.limit)
    else:
        ap.error("pass --dataset, or --labels plus --dir")
    if args.limit:
        dataset = dataset[:args.limit]

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else [args.layer]
    reader = HiddenReader(allow_download=args.allow_download)
    feats: dict[tuple[int, str], list[np.ndarray]] = {
        (l, p): [] for l in layers for p in pools}
    for i, d in enumerate(dataset, 1):
        grid = reader.read_grid(d["context"], d["question"], tuple(layers),
                                tuple(pools))
        for key, vec in grid.items():
            feats[key].append(vec)
        if i % 10 == 0:
            print(f"  extracted {i}/{len(dataset)}", flush=True)

    table: dict[str, float] = {}
    chance: dict[str, float] = {}
    y = np.array([d["label"] for d in dataset])
    for key in feats:
        x = np.array(feats[key])
        name = f"L{key[0]}/{key[1]}"
        table[name] = cv_auroc(x, y, folds=args.folds, l2=args.l2, pca=args.pca or None)
        # Same pipeline on shuffled labels: with n this small an AUROC of 0.7 can be
        # ordinary noise, and this is the only honest way to say so.
        perm = [cv_auroc(x, np.random.default_rng(s).permutation(y),
                         folds=args.folds, l2=args.l2, pca=args.pca or None)
                for s in range(args.permutations)]
        chance[name] = float(np.mean(perm)) + float(np.std(perm))
        print(f"{name:<14} CV-AUROC {table[name]:.3f}   "
              f"chance+1sd {chance[name]:.3f}   "
              f"(n={len(y)}, pos={int(y.sum())}, pca={args.pca})", flush=True)

    print(f"\nlayer {args.layer}, {args.folds}-fold CV, L2={args.l2}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"layer": args.layer, "folds": args.folds, "l2": args.l2,
                   "pca": args.pca, "n": len(dataset), "auroc": table,
                   "chance": chance}, f, indent=2)
    print(f"wrote {args.out}")
    print("NOTE: n is small. Report the number and its n, never the number alone.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
