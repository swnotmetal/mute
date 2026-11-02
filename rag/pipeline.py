"""Wire the RAG loop together and expose a tiny CLI demo."""

import argparse
from pathlib import Path

from .chunker import Chunk, chunk_text
from .config import RagConfig
from .generator import Generator
from .loaders import load_dir
from .retrieval import Embedder, HybridIndex, VectorIndex

PROMPT_TEMPLATE = """Use ONLY the context below to answer the question. If the answer is not
present in the context, reply exactly "资料中没有相关信息" and do not guess.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self.embedder: Embedder | None = None
        self.index: VectorIndex | None = None
        self.generator: Generator | None = None

    def build_index(self, doc_dir: str) -> dict:
        docs, skipped = load_dir(doc_dir)
        chunks: list[Chunk] = []
        for d in docs:
            fname = Path(d.source).name
            for i, piece in enumerate(chunk_text(d.text, self.cfg.chunk_size, self.cfg.chunk_overlap)):
                # Prepend the filename so descriptive names ("Phishing-Email-Practise",
                # "Simple car rental") help retrieval even when the word is not in the body.
                text = f"[file: {fname}]\n{piece}"
                chunks.append(Chunk(text=text, source=d.source, index=i))

        self.embedder = Embedder(self.cfg.embed_model, max_length=self.cfg.embed_max_length)
        vectors = self.embedder.encode([c.text for c in chunks], batch_size=self.cfg.embed_batch)
        self.index = HybridIndex(vectors, chunks)
        return {"documents": len(docs), "chunks": len(chunks), "skipped": skipped}

    def retrieve(self, question: str, k: int | None = None):
        k = k or self.cfg.top_k
        if self.index is None:
            raise RuntimeError("call build_index() first")
        qvec = self.embedder.encode([question], is_query=True)[0]
        return self.index.search(qvec, question, k=k)

    def answer(self, question: str, k: int | None = None):
        hits = self.retrieve(question, k)
        context = "\n\n".join(f"[{i + 1}] ({c.source})\n{c.text}" for i, (c, _) in enumerate(hits))
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        if self.generator is None:
            self.generator = Generator(self.cfg.gen_model, self.cfg.max_new_tokens, self.cfg.load_in_4bit)
        return self.generator.answer(prompt), hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal RAG demo")
    ap.add_argument("--dir", required=True, help="document directory to index")
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gen", default=None, help="override generator model id")
    args = ap.parse_args()

    cfg = RagConfig()
    if args.gen:
        cfg.gen_model = args.gen

    pipe = RAGPipeline(cfg)
    stats = pipe.build_index(args.dir)
    print(f"indexed: {stats['documents']} documents, {stats['chunks']} chunks, "
          f"{len(stats['skipped'])} skipped")
    for path, reason in stats["skipped"][:10]:
        print(f"  skip: {path} ({reason})")

    answer, hits = pipe.answer(args.question, k=args.k)
    print("\n=== ANSWER ===\n" + answer)
    print("\n=== SOURCES ===")
    for c, score in hits:
        print(f"  [{score:.3f}] {c.source} (chunk {c.index})")


if __name__ == "__main__":
    main()
