# MUTE

A small, local experiment on one question: **can a language model tell, before it
writes anything, that it should not answer?**

The repository started as a linear probe on hidden states (the notebook), and was
then pushed through three harder settings: a document RAG pipeline, a forced
extraction task, and a set of questions that assert something false. Each setting
produced a number. Most of them are negative results.

Everything runs locally. No API keys, no hosted models.

## What is measured

| # | Experiment | Data | n | Result |
| --- | --- | --- | --- | --- |
| 1 | Missing-context detection from hidden states | 300 CUAD contract questions, 150 answerable / 150 not | 300 | **AUROC 0.950** (95% CI 0.882–0.993) |
| 2 | The same question on a document RAG pipeline | 20 hand-labelled questions over 121 personal documents | 17 | **9/17 = 53%**; fact questions 8/10 |
| 3 | Forced extraction of absent fields | 51 fields, 48 verified absent from the collection | 46 | **18/46 = 39%** fabrication |
| 4 | Questions that assert a false premise | 42 items, 5 adversarial families + controls | 40 | **13/40 unsupported answers**, 21 correct refusals |

## 1. Missing-context detection (the original claim)

A linear probe on Qwen3.5-0.8B hidden states, trained to separate "the answer is in
this window" from "it is not". Layer sweep over all 24 layers; best is layer 14
(0.946 five-fold CV on the train split), 0.950 blind test.

A position-shortcut check randomises where the answer sits inside the window:
AUROC stays at 0.935 / AUPRC 0.910, so the probe reads presence, not position.

This is the project's one positive result. It is also the least original: see
*Prior work*.

## 2. The same idea on a document RAG pipeline

Retrieval (keyword + dense, rank-fused), cross-encoder reranking with a
relative score gate, then a local generator told to stay inside the retrieved text.

Answer quality on 17 usable questions: **53%**, fact questions **80%**.

The informative part is *why* the other 8 failed. Of them:

- 5–6 are **upstream**: the answer either never reached the model (evidence cut by
  the per-chunk character budget) or the right document was never retrieved;
- 1 is a grading error (the answer names the right challenges, the rule asked for
  three specific names);
- 1 is model-side.

So on this pipeline the probe has almost nothing to detect. The failure happens
before the model speaks.

## 3. Substitution under forced answers

A tabular set built from a private grade sheet: 100 bait items (the asked row is
removed, other real values stay) and 100 matched controls. Entity names are nonce
codes; a leak check confirms **0/100** items can be answered without the context.

Under a prompt that forbids abstaining:

| Behaviour | Count |
| --- | --- |
| abstained anyway | 94/100 |
| substituted another row's value | 3/100 |
| interpolated a value from the table | 3/100 |
| controls answered correctly | 99/100 |

The model refuses far more than it invents, and it can read the table perfectly when
the row is there. **Absence that is visible is not an invitation to fabricate.**

## 4. Fabrication from false premises

42 items over an inventory sheet, in five families, graded by string rules with no
model in the loop: *count* (assert a wrong number), *ranking* (ask for a ranking the
sheet has no column for), *restock* (ask for a date or supplier that is not stored),
*variant* (ask about a colour that does not exist), *sycophancy* (state something
false and ask for confirmation).

| Family | Fabricated |
| --- | --- |
| ranking | **5/7** |
| count | 3/7 |
| restock | 1/7 |
| variant | 1/6 |
| sycophancy | 1/7 |

The failure is not random invention. The model **reinterprets a real column and then
reasons correctly from the wrong premise** — it read `Reorder Level` as a sales
proxy, `Status` as an out-of-stock flag, and answered "stock: 0" for a SKU that does
not exist.

Cheap signals on the same 40 items (13 positive, 27 negative):

| Signal | Available | AUROC |
| --- | --- | --- |
| generic structural checker | before generation | **0.516** |
| model self-assessment | before generation | **0.376** |
| 1 − HHEM-2.1 entailment score | after the answer | 0.741 |
| answer length | after the answer | 0.781 |

The two signals that work need the answer first, so they filter rather than gate.

## What is established

- Visible absence suppresses fabrication: 94/100 refusals with five same-format
  values on screen, even when abstaining is forbidden.
- On a raw document directory, pipeline failures outnumber model failures about 6:1.
- Ranking and count questions elicit fabrication; existence and confirmation checks
  mostly do not.
- A 0.1B entailment model already covers most answer-support checking (83% recall,
  0/3 false alarms on the earlier set).

## What is not established

- Whether a hidden-state probe adds anything on this kind of failure. In a 40-item
  pilot the probe's best cell was 0.732 (L18, question-token pooling) against a
  noise band of about 0.52–0.60. At n = 40 the 95% interval on any AUROC here is
  ±0.19, so nothing in that table is separated from anything else.
- Token-position pooling as a fix. Pooling over digit tokens scored 0.527, i.e.
  chance; mean-pooling over the whole prompt scored 0.689, the same as the last
  token. The "the signal lives at the value tokens" hypothesis is not supported.

## Falsified during the work

Recorded because they were argued for and then measured:

1. A generic 30-line structural checker catches fabrication. It flagged 33/40 items
   at precision 0.33; AUROC 0.516.
2. Fabrications are built from real values, so entailment models cannot see them.
   HHEM flags them strongly — inverted, its AUROC is 0.741.
3. Absence alone drives fabrication. It needs absence that is *invisible*: with a
   small table on screen and the asked row visibly missing, 94% of answers refuse.

## Prior work

The two central ideas here are not new, and the repository is a replication with
different data rather than a new method.

- **Corrective RAG** (Yan et al., arXiv 2401.15884, 2024) adds a lightweight
  retrieval evaluator that scores retrieved documents and triggers corrective
  actions. That is the same pre-generation gate this project tests.
- **Self-RAG** (Asai et al., arXiv 2310.11511, 2023) learns reflection tokens to
  decide when to retrieve and to critique its own output.
- **Semantic Entropy Probes** (Kossen et al., arXiv 2406.15927, 2024) approximate
  semantic entropy from hidden states of a single generation, and their ablations
  cover exactly the layer-and-token-position question asked here.
- **INSIDE** (Chen et al., arXiv 2402.03744, ICLR 2024) shows internal states carry
  hallucination signal.
- **ParamMute** (arXiv 2502.15543) identifies mid-to-deep FFNs that dominate when
  parametric knowledge overrides retrieved evidence — the mechanism behind
  substitution errors.
- **BELIEFRAG** (arXiv 2609.39139, 2026) drives retrieve / rewrite / verify / answer /
  abstain from an explicit evidence state, and reports that calibrated
  answerability is the strongest operational component.

What is specific to this repository is the measurement discipline, not the method:
deterministic grading, published class counts, permutation baselines, and a list of
the traps that produced wrong numbers before they were caught (grading bugs, an 8%
label error rate in the absent-field set, parametric leakage in a constructed set,
value-space degeneracy, token truncation mistaken for refusal, and picking the best
of twelve correlated tests).

## Repository layout

```
llm_client.py         one HTTP path for both local backends
mute_structured.py    field schema, context builder, JSON extraction helpers
grounding.py          HHEM-2.1-Open wrapper
mute_hidden.py        hidden-state reader; layer / pooling sweep; ridge probe
adversarial_set.py    42 items + deterministic grading rules
mute_adversarial.py   runs them, writes per-item diagnostics
mute_baselines.py     cheap-signal scoring and AUROC comparison
make_substitution_set.py  builds the bait/control set from a private table
mute_compare.py       structured extraction vs cheap signals
qa_eval.py            20-question document eval with hand-written rules
probe_dataset.py      turns a mute_compare run into probe rows
rag_pro/              the document pipeline (parse, retrieve, rerank, answer)
```

## Reproduce

Local model server (llama.cpp, OpenAI-compatible) on `127.0.0.1:8080`; only the
embedding model needs Ollama.

```bash
python mute_adversarial.py --condition plain --max-tokens 6000
python mute_adversarial.py --report probe_runs/adv_plain.jsonl
python mute_baselines.py --rows probe_runs/adv_plain.jsonl --emit-probe probe_runs/adv_probe.jsonl
python mute_hidden.py --dataset probe_runs/adv_probe.jsonl --layers 4,8,12,15,18,21 --pools last,question
python -m rag_pro.qwen_rag --dir "PATH/TO/DOCUMENTS" --question "..."
python qa_eval.py --run --dir "PATH/TO/DOCUMENTS" && python qa_eval.py --report
```

Long runs write one JSON line per item and resume where they stopped.

## What this repository does not claim

- It is not a hallucination detector, and the probe is not wired into the RAG path.
- It does not show that hidden-state probes beat retrieval scores, self-assessment,
  or entailment models. On the data here, none of those comparisons are resolved.
- The document results come from one 121-file personal collection, and the
  adversarial results from one 27-row sheet. Neither generalises on its own.
- Layer 15 (or 23) is not universally optimal, and any single-layer number here is
  one draw from a 24-layer sweep.
