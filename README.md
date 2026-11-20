# Retrieval-Augmented-Generation

[![Boot.dev Learn Retrieval Augmented Generation certificate](https://api.boot.dev/v1/static/certificate/aaf8c834-eeda-48d7-8203-b2230532f6b3)](https://www.boot.dev/certificates/aaf8c834-eeda-48d7-8203-b2230532f6b3)

End-to-end Retrieval-Augmented Generation (RAG) search engine and CLI suite built for **Webflyx** as part of Boot.dev's *Learn Retrieval Augmented Generation* course[cite: 1]. Each folder captures a progressive milestone—from lexical search and dense embeddings to hybrid ranking with Reciprocal Rank Fusion (RRF), LLM-assisted query enhancement and evaluation, grounded generation, and CLIP multimodal search[cite: 1].

---

## Repository Structure & Progression Map

### Top-Level Layout
- `Keyword vs. Semantic Search/` — Foundations of lexical/BM25 keyword retrieval and dense semantic/chunked search[cite: 1].
- `hybrid with ranking/` — Combined lexical and semantic retrieval with weighted scoring and Reciprocal Rank Fusion (RRF)[cite: 1].
- `llm-chunking and judge/` — LLM-driven query enhancement, reranking, precision/recall/F1 evaluation, debug logging, and LLM-as-a-judge scoring[cite: 1].
- `rag/` — End-to-end RAG generation workflows (summarization, inline citations, multimodal image+text prompting, and CLIP image-to-movie encoding/search)[cite: 1].
- `.gitignore` — Excludes OS metadata, Python cache/virtual environments, and large binary/cache artifacts (`.npy`, `.pkl`, cache folders)[cite: 1].
- `README.md` — Repository overview, folder map, and usage guide[cite: 1].

---

### 1. `Keyword vs. Semantic Search/`
Covers lexical keyword search mechanics (tokenization, stopword filtering, stemming, TF-IDF, BM25 IDF/TF/length normalization) followed by semantic embeddings and chunked vector search[cite: 1].

- `rag-without-semantic/` — Step-by-step lexical/BM25 search progression across 13 numbered stages (`1/` through `13 - bm scores with 5 results/`)[cite: 1].
- `semantic search/` — Dense embedding and chunking progression[cite: 1]:
  - `1 - semantic- using sentence_transformers model and verify_embeddings/` — Generates sentence embeddings and verifies embedding shapes on `movies.json`[cite: 1].
  - `4- semantic chunking/` — Implements semantic chunking over movie text[cite: 1].
  - `5- chunk with embedding/` — Generates embeddings across text chunks[cite: 1].
  - `6 - chunk with semantic embedding/` — Builds semantic chunk embeddings and `chunk_metadata.json`[cite: 1].
  - `7- Chunked Semantic Search/` — Performs retrieval over chunked movie embeddings[cite: 1].

---

### 2. `hybrid with ranking/`
Combines keyword and semantic retrieval streams and merges rankings across result sets[cite: 1].

- `1- weighted search hybrid search - optinal/` — Explores weighted hybrid scoring combining keyword and semantic signals[cite: 1].
- `2- hybrid search with limit/` — Adds result limiting/top-k handling to hybrid search[cite: 1].
- `3- HybridSearch.rrf_search/` — Implements `HybridSearch.rrf_search` using Reciprocal Rank Fusion[cite: 1].

---

### 3. `llm-chunking and judge/`
Integrates LLM calls for query cleanup, candidate reranking, quantitative retrieval evaluation, and automated judging[cite: 1].

- `1- spell fix using llm open router/` — Fixes spelling/typos in user search queries using an LLM via OpenRouter[cite: 1].
- `2- llm for batch ranking/` — Uses an LLM to rank batches of retrieved movie candidates[cite: 1].
- `3- llm for reranking using one llm call /` — Performs candidate reranking in a single consolidated LLM call[cite: 1].
- `4- llm with precision and recall/` — Evaluates retrieval quality against `golden_dataset.json` using precision and recall[cite: 1].
- `5- llm with f1-score/` — Extends evaluation metrics with F1 score[cite: 1].
- `6- with debug and log the process in llm/` — Adds step-by-step debugging and logging to LLM pipeline stages[cite: 1].
- `7- llm as a judge instead of manuall debug/` — Replaces manual inspection with LLM-as-a-judge relevance grading[cite: 1].

---

### 4. `rag/`
Connects retrieved context and multimodal inputs to grounded generation and CLIP similarity search[cite: 1].

- `1- rag/` — Core end-to-end RAG CLI (`augmented_generation_cli.py`) answering queries from retrieved movies[cite: 1].
- `2- rag summerize search/` — Synthesizes multi-document summaries across retrieved search results[cite: 1].
- `3- rag with citations position arguments on movie dataset/` — Generates responses with inline source citations using positional CLI arguments[cite: 1].
- `5- image rag/` — Adds `describe_image_cli.py` and `multimodal_search.py` to combine image + text context for search/generation[cite: 1].
- `7- image encoding and embeding using progress bar/` — Adds `multimodal_search_cli.py` with `verify-image-embedding` and `image-search` using CLIP (`clip-ViT-B-32`), Pillow, and progress-bar text embedding encoding[cite: 1].

---

### Typical Step Folder Anatomy
Each numbered step directory follows a consistent layout[cite: 1]:
- `cli/` — Executable CLI entry points (`keyword_search_cli.py`, `semantic_search_cli.py`, `hybrid_search_cli.py`, `evaluation_cli.py`, `augmented_generation_cli.py`, `describe_image_cli.py`, `multimodal_search_cli.py`, `test_llm.py`)[cite: 1].
- `cli/lib/` — Core modules (`keyword_search.py`, `semantic_search.py`, `hybrid_search.py`, `query_enhancement.py`, `reranking.py`, `evaluation.py`, `multimodal_search.py`, `search_utils.py`)[cite: 1].
- `data/` — Source datasets (`movies.json`, `stopwords.txt`, `golden_dataset.json`) and sample images (`paddington.jpeg`)[cite: 1].
- `cache/` — Metadata and runtime index/embedding caches (`chunk_metadata.json`, while large `.npy`/`.pkl` artifacts are gitignored)[cite: 1].
- `pyproject.toml` & `uv.lock` — Python dependency specification and `uv` lockfile[cite: 1].
- `README.md` — Step-specific notes[cite: 1].

---

## Running a Step

Navigate into any step directory (for example, the final multimodal RAG step) and run commands with `uv`[cite: 1]:

```bash
cd "rag/7- image encoding and embeding using progress bar"

# Verify CLIP image embedding dimensions
uv run cli/multimodal_search_cli.py verify-image-embedding "data/paddington.jpeg"

# Search movies by image similarity
uv run cli/multimodal_search_cli.py image-search "data/paddington.jpeg"
