"""Local Qwen generator (works for Qwen3.5 multimodal checkpoints and Qwen2.5 causal LMs)."""

import torch
from transformers import AutoTokenizer


class Generator:
    def __init__(self, model_id: str, max_new_tokens: int = 256, load_in_4bit: bool = False):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = _load_model(model_id, load_in_4bit)
        self.device = self.model.device

    def answer(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _load_model(model_id: str, load_in_4bit: bool):
    kwargs: dict = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16

    # Qwen3.5 ships as Qwen3_5ForConditionalGeneration; AutoModelForCausalLM does not
    # cover it, so fall back to the explicit class.
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **kwargs)
