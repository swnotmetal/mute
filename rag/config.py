from dataclasses import dataclass


@dataclass
class RagConfig:
    # Embedding: small, fast, Chinese-friendly (~95 MB). Swap to BAAI/bge-m3 for
    # multilingual + long-context in phase 2.
    embed_model: str = "BAAI/bge-small-zh-v1.5"

    # Generator: reuse the already-cached Qwen3.5-0.8B to validate the pipeline.
    # Swap to "Qwen/Qwen2.5-7B-Instruct" (optionally 4-bit) for the real run.
    gen_model: str = "Qwen/Qwen3.5-0.8B"
    load_in_4bit: bool = False

    chunk_size: int = 500      # characters (works for zh + en)
    chunk_overlap: int = 80
    top_k: int = 5
    max_new_tokens: int = 256
    embed_batch: int = 32
    embed_max_length: int = 512
