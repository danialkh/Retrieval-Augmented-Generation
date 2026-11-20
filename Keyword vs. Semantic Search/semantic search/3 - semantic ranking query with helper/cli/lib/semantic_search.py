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