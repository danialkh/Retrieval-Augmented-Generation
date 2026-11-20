import json
import os
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_CACHE_PATH = os.path.join("cache", "movie_embeddings.npy")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def _index_documents(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

    def build_embeddings(self, documents: list[dict]):
        self._index_documents(documents)

        movie_texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)

        cache_dir = os.path.dirname(EMBEDDINGS_CACHE_PATH)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(EMBEDDINGS_CACHE_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self._index_documents(documents)

        if os.path.exists(EMBEDDINGS_CACHE_PATH):
            self.embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        scored_docs: list[tuple[float, dict]] = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)

        results: list[dict] = []
        for score, doc in scored_docs[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results

    
def _load_movies() -> list[dict[str, Any]]:
    for candidate in ("data/movies.json", "movies.json"):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                data: Any = json.load(f)

            if isinstance(data, dict) and isinstance(data.get("movies"), list):
                return data["movies"]
            if isinstance(data, list):
                return data
            raise ValueError(f"Unexpected JSON structure in {candidate}.")

    raise FileNotFoundError("Could not find movies.json.")

def verify_model() -> None:
    searcher = SemanticSearch()
    print(f"Model loaded: {searcher.model}")
    print(f"Max sequence length: {searcher.model.max_seq_length}")

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks

import re




def embed_text(text: str) -> None:
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    searcher = SemanticSearch()
    documents = _load_movies()
    embeddings = searcher.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def semantic_search(query: str, limit: int = 5) -> None:
    searcher = SemanticSearch()
    documents = _load_movies()
    searcher.load_or_create_embeddings(documents)
    results = searcher.search(query, limit=limit)

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description']}")


from typing import Any, TypedDict

CHUNK_EMBEDDINGS_CACHE_PATH = os.path.join("cache", "chunk_embeddings.npy")
CHUNK_METADATA_CACHE_PATH = os.path.join("cache", "chunk_metadata.json")


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self._index_documents(documents)

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for doc_idx, doc in enumerate(documents):
            description = (doc.get("description") or "").strip()
            if not description:
                continue

            doc_chunks = semantic_chunk_text(
                description, max_chunk_size=4, overlap=1
            )
            total_chunks = len(doc_chunks)

            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        embeddings: np.ndarray = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_embeddings = embeddings
        self.chunk_metadata = chunk_metadata

        cache_dir = os.path.dirname(CHUNK_EMBEDDINGS_CACHE_PATH)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        np.save(CHUNK_EMBEDDINGS_CACHE_PATH, embeddings)
        with open(CHUNK_METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return embeddings

    def load_or_create_chunk_embeddings(
        self, documents: list[dict[str, Any]]
    ) -> np.ndarray:
        self._index_documents(documents)

        if os.path.exists(CHUNK_EMBEDDINGS_CACHE_PATH) and os.path.exists(
            CHUNK_METADATA_CACHE_PATH
        ):
            embeddings: np.ndarray = np.load(CHUNK_EMBEDDINGS_CACHE_PATH)
            with open(CHUNK_METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
                metadata_payload = json.load(f)
            self.chunk_embeddings = embeddings
            self.chunk_metadata = metadata_payload.get("chunks", [])
            return embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if (
            self.chunk_embeddings is None
            or self.chunk_metadata is None
            or self.documents is None
        ):
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict[str, Any]] = []
        for chunk_embedding, meta in zip(self.chunk_embeddings, self.chunk_metadata):
            score = float(cosine_similarity(query_embedding, chunk_embedding))
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": score,
                }
            )

        best_movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in best_movie_scores or score > best_movie_scores[movie_idx]:
                best_movie_scores[movie_idx] = score

        ranked_movies = sorted(
            best_movie_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]

        results: list[dict[str, Any]] = []
        for movie_idx, score in ranked_movies:
            doc = self.documents[movie_idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": (doc.get("description") or "")[:100],
                    "score": round(score, 3),
                }
            )

        return results


import re


def semantic_chunk_text(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(raw_sentences) == 1 and not raw_sentences[0].endswith((".", "!", "?")):
        sentences = [cleaned]
    else:
        sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    step = max_chunk_size - overlap
    start = 0

    while start < len(sentences):
        chunk = " ".join(s.strip() for s in sentences[start : start + max_chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chunk_size >= len(sentences):
            break
        start += step

    return chunks

def search_chunks(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if (
            self.chunk_embeddings is None
            or self.chunk_metadata is None
            or self.documents is None
        ):
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict[str, Any]] = []
        for chunk_embedding, meta in zip(self.chunk_embeddings, self.chunk_metadata):
            score = float(cosine_similarity(query_embedding, chunk_embedding))
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": score,
                }
            )

        best_movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in best_movie_scores or score > best_movie_scores[movie_idx]:
                best_movie_scores[movie_idx] = score

        ranked_movies = sorted(
            best_movie_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]

        results: list[dict[str, Any]] = []
        for movie_idx, score in ranked_movies:
            doc = self.documents[movie_idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": (doc.get("description") or "")[:100],
                    "score": round(score, 3),
                }
            )

        return results

def embed_chunks() -> None:
    documents = _load_movies()
    searcher = ChunkedSemanticSearch()
    embeddings = searcher.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")