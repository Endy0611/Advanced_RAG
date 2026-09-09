# 3-Stage RAG Pipeline

A minimal, local retrieval-augmented generation pipeline: embed → retrieve → rerank → generate — all running on local models via [Ollama](https://ollama.com).

## Pipeline

1. **Retrieve** — embed the corpus + query with `nomic-embed-text`, rank by cosine similarity, keep top `TOP_K`.
2. **Rerank** — rescore those candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`, CPU) for higher precision, keep top `TOP_N`.
3. **Generate** — build a context-grounded prompt from the final chunks and answer with `qwen3:4b`.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

## Setup

```bash
pip install ollama numpy sentence-transformers --break-system-packages

ollama pull nomic-embed-text
ollama pull qwen3:4b
```

The reranker model downloads automatically from Hugging Face on first run.

## Usage

```bash
python3 rag_pipeline.py "How much vacation do I get each year?"

# or run with no args to use the default sample query
python3 rag_pipeline.py
```

Each run prints all three stages: retrieved candidates with similarity scores, reranked candidates with rerank scores, then the final generated answer.

## Configuration

Edit the constants at the top of `rag_pipeline.py`:

| Constant | Purpose |
|---|---|
| `EMBED_MODEL` | Ollama embedding model |
| `GEN_MODEL` | Ollama chat/generation model |
| `RERANK_MODEL` | Hugging Face cross-encoder model |
| `TOP_K` | Candidates kept after retrieval |
| `TOP_N` | Candidates kept after reranking |
| `CORPUS` | List of `(source_file, text)` tuples — swap in your own documents here |

## Notes

- `CORPUS` is a hardcoded sample knowledge base (HR-policy style docs). Replace it with your own chunked documents to use this on real data.
- No vector DB — similarity search is done in-memory with numpy, fine for small corpora. For larger corpora, swap `stage1_retrieve` for a proper vector store (e.g. ChromaDB).