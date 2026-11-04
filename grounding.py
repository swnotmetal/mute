"""Claim grounding via Vectara HHEM-2.1-Open (0.1B, CPU, Apache-2.0).

Premise/hypothesis entailment: 0 = not supported by the premise, 1 = fully
supported. Replaces the hand-rolled substring check as the "is this value in the
retrieved context" signal.

shortcut: HHEM-2.1-Open is English-trained. The corpus is ~94% English (6 of 101
documents are German/Finnish course files), so those may score poorly. Upgrade to
a multilingual checker if that subset matters.
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification

MODEL_ID = "vectara/hallucination_evaluation_model"


class Supporter:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True)

    def score(self, premise: str, hypotheses: list[str]) -> list[float]:
        if not hypotheses:
            return []
        pairs = [(premise, h) for h in hypotheses]
        return [float(x) for x in self.model.predict(pairs)]
