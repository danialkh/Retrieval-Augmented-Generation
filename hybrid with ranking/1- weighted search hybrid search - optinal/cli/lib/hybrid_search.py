import os
from typing import Any

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in scores]


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self, query: str, limit: int
    ) -> list[tuple[dict[str, Any], float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(
        self, query: str, k: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


    @staticmethod
    def _combine_scores(bm25_score: float, semantic_score: float, alpha: float) -> float:
        return alpha * semantic_score + (1.0 - alpha) * bm25_score

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[dict[str, Any]]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        search_limit = limit * 10
        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, search_limit)

        bm25_by_id: dict[int | str, tuple[dict[str, Any], float]] = {
            doc["id"]: (doc, score) for doc, score in bm25_results
        }
        semantic_by_id: dict[int | str, tuple[dict[str, Any], float]] = {
            item["id"]: (item, item["score"]) for item in semantic_results
        }

        candidate_ids = list(dict.fromkeys([*bm25_by_id.keys(), *semantic_by_id.keys()]))
        if not candidate_ids:
            return []

        raw_bm25 = [
            bm25_by_id[doc_id][1] if doc_id in bm25_by_id else 0.0
            for doc_id in candidate_ids
        ]
        raw_semantic = [
            semantic_by_id[doc_id][1] if doc_id in semantic_by_id else 0.0
            for doc_id in candidate_ids
        ]

        norm_bm25 = normalize_scores(raw_bm25)
        norm_semantic = normalize_scores(raw_semantic)

        combined_results: list[dict[str, Any]] = []
        for idx, doc_id in enumerate(candidate_ids):
            bm25_val = norm_bm25[idx]
            sem_val = norm_semantic[idx]
            hybrid_score = self._combine_scores(bm25_val, sem_val, alpha)

            if doc_id in bm25_by_id:
                doc = bm25_by_id[doc_id][0]
                title = doc["title"]
                description = doc.get("description") or ""
            else:
                sem_doc = semantic_by_id[doc_id][0]
                title = sem_doc["title"]
                description = sem_doc.get("document") or ""

            combined_results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "document": description,
                    "hybrid_score": hybrid_score,
                    "bm25_score": bm25_val,
                    "semantic_score": sem_val,
                }
            )

        combined_results.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return combined_results[:limit]