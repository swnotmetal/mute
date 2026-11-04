# RAG prototype

A small local question-answering system. It reads a folder of documents, finds
the parts that look relevant to a question, and asks a local language model to
answer from those parts only.

The folder is a parameter, so nothing here depends on one particular document
set. No document names, people, organisations or dates are used in this file.

## The flow

```
documents -> parse -> chunks -> search (two ways) -> merge by rank
          -> rerank -> keep the relevant ones -> answer -> check the answer
```

## What each step does

| Step | Plain description | Uses |
| --- | --- | --- |
| Parse | Turn PDF, DOCX, PPTX, XLSX, CSV and TXT into text with page and chunk ids. Done once, then cached. | Qwen-Agent `DocParser` |
| Chunk | Split each document into small pieces (about one page each). | parser settings |
| Search, keyword | Classic word matching. Good with names and file names, weak with paraphrases. | BM25 (`rank_bm25`) |
| Search, meaning | Turn text into vectors and compare meanings. Works across languages. | `bge-m3` via Ollama |
| Merge | Combine the two rankings by position, not by raw score, so the two scales never have to match. | RRF |
| Rerank | Score each candidate against the question directly and keep the best. | `bge-reranker-v2-m3` |
| Keep | Drop candidates that score far below the best one. | relative cut |
| Answer | The model sees only the kept text and must stay inside it. | local model via Ollama or llama.cpp |
| Check | Compare the answer with the retrieved text. | HHEM-2.1-Open or a simple check |

## The settings that matter

| Setting | Default | What it controls |
| --- | --- | --- |
| `--retrieve-k` | 20 | how many candidates are pulled before reranking |
| `max_per_doc` | 2 | how many chunks one document may contribute |
| `expand_docs` / `expand_per_doc` | 3 / 3 | how many extra chunks are pulled from the strongest documents |
| rerank ratio | 0.15 | keep chunks scoring at least 15% of the best score |
| `--top-k` | 5 | how many chunks the model actually reads |

## Design choices, and why

- **Two search methods, not one.** Word matching fails when the question uses
  different words from the document; meaning search fails on exact names. They
  fail in different places, so they are combined.
- **Merge by rank, not by score.** Keyword scores and vector scores live on
  different scales. Comparing their order avoids normalising them.
- **Two stages.** First stage casts a wide net. Second stage scores each
  candidate against the question and throws most of them away. A single stage
  cannot be both wide and precise.
- **Relative cut, not a fixed one.** A fixed cut either removes weak but useful
  text, or keeps strong but useless text. Cutting relative to the best score
  adapts to each question.
- **One document cannot take over.** Without a cap, one long file fills the
  whole candidate list and other files never get a chance.
- **Add neighbouring text when a document is clearly relevant.** Search often
  returns only a front page or a table of contents. Pulling more of that same
  document is what puts the actual answer in front of the model.
- **Multilingual models where the corpus is mixed.** A model that only knows
  one language silently scores correct text as irrelevant.

## Problems found during development (each is a general lesson)

1. **Tables inside office documents disappear** if you read only paragraphs.
   Use a parser that reads table cells.
2. **Some PDFs have no text layer** (scans). They parse into empty chunks and
   then waste a slot in the context. Retry with a second PDF reader; if it is
   still empty, skip the file and print a warning. Never keep silent blank
   chunks.
3. **A file's first chunk is often a cover page.** If search only ever returns
   first chunks, the answer text never arrives. Expand within the document.
4. **One large file can occupy most of the candidate list.** Cap it per
   document.
5. **A small reranker can be too strict.** Used with a hard cut-off it removes
   correct text. Use it to rank, and cut relative to the best score.
6. **Keyword matching needs word boundaries.** A short term must not match
   inside a longer word.
7. **A model that never refuses will invent values.** With no option to say
   "not in the documents", it fills missing fields with plausible numbers.
   That behaviour belongs to the model, not to the retriever.

## Evaluation

A fixed list of questions with expected key terms. A question passes when the
answer is not a refusal and contains at least one expected term. The pass bar
is 75%.

Current state: 6 of 8 questions pass on a small hand-built set. The two
failures are understood: one asks for information the documents do not contain,
and one needs a fact that sits deep inside a document that search never reached.

A refusal is not the same as a good answer. Counting refusals as passes is the
easiest way to fool yourself.

## Running it

```bash
python -m rag_pro.qwen_rag --dir "PATH/TO/DOCUMENTS" --question "Your question"
python -m rag_pro.eval_rag --dir "PATH/TO/DOCUMENTS"
```

Optional: `--no-rerank` to skip the second stage, `--top-k` and
`--retrieve-k` to change how much text is used.

## Limits

- Quality is capped by the text layer of the source files.
- Counting and cross-file aggregation questions are better answered from a
  metadata table than from a retriever. Vector search does not count.
- Images, diagrams, spreadsheets of pictures and video cannot be read at all.
- The generator can still add unsupported detail. Check the answer against the
  retrieved text before trusting it.
