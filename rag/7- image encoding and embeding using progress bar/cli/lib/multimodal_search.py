from typing import Any

import numpy as np
from lib.semantic_search import _load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MultimodalSearch:
    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict[str, Any]] = documents or []
        self.texts: list[str] = [
            f"{doc.get('title', '')}: {doc.get('description', doc.get('document', ''))}"
            for doc in self.documents
        ]
        if self.texts:
            self.text_embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True,
            )
        else:
            self.text_embeddings = np.empty((0, 512), dtype=np.float32)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict[str, Any]]:
        image_embedding = self.embed_image(image_path)
        scored_results: list[dict[str, Any]] = []

        for idx, doc in enumerate(self.documents):
            text_emb = self.text_embeddings[idx]
            similarity = _cosine_similarity(image_embedding, text_emb)
            doc_id = doc.get("id", idx)
            title = doc.get("title", "")
            description = doc.get("description", doc.get("document", ""))
            scored_results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "description": description,
                    "similarity": similarity,
                }
            )

        scored_results.sort(key=lambda item: item["similarity"], reverse=True)
        return scored_results[:limit]


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict[str, Any]]:
    documents = _load_movies()
    searcher = MultimodalSearch(documents=documents)
    return searcher.search_with_image(image_path, limit=5)