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


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)
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

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(
        self, query: str, alpha: float = 0.5, limit: int = 5
    ) -> list[dict[str, Any]]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self._semantic_search(query, fetch_limit)

        bm25_by_id: dict[int | str, tuple[dict[str, Any], float]] = {
            doc["id"]: (doc, score) for doc, score in bm25_results
        }
        semantic_by_id: dict[int | str, tuple[dict[str, Any], float]] = {
            item["id"]: (item, item["score"]) for item in semantic_results
        }

        doc_ids = list(dict.fromkeys([*bm25_by_id.keys(), *semantic_by_id.keys()]))
        if not doc_ids:
            return []

        raw_bm25 = [
            bm25_by_id[doc_id][1] if doc_id in bm25_by_id else 0.0
            for doc_id in doc_ids
        ]
        raw_semantic = [
            semantic_by_id[doc_id][1] if doc_id in semantic_by_id else 0.0
            for doc_id in doc_ids
        ]

        norm_bm25 = normalize_scores(raw_bm25)
        norm_semantic = normalize_scores(raw_semantic)

        combined_by_id: dict[int | str, dict[str, Any]] = {}
        for idx, doc_id in enumerate(doc_ids):
            b_score = norm_bm25[idx]
            s_score = norm_semantic[idx]
            h_score = hybrid_score(b_score, s_score, alpha)

            if doc_id in bm25_by_id:
                base_doc = bm25_by_id[doc_id][0]
                title = base_doc["title"]
                description = base_doc.get("description") or ""
            else:
                base_doc = semantic_by_id[doc_id][0]
                title = base_doc["title"]
                description = base_doc.get("document") or ""

            combined_by_id[doc_id] = {
                "id": doc_id,
                "title": title,
                "document": description,
                "bm25_score": b_score,
                "semantic_score": s_score,
                "hybrid_score": h_score,
            }

        ranked = sorted(
            combined_by_id.values(),
            key=lambda entry: entry["hybrid_score"],
            reverse=True,
        )
        return ranked[:limit]

    def rrf_search(
        self, query: str, k: int = 60, limit: int = 10
    ) -> list[dict[str, Any]]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self._semantic_search(query, fetch_limit)

        doc_entries: dict[int | str, dict[str, Any]] = {}

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            doc_entries[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc.get("description") or "",
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, item in enumerate(semantic_results, start=1):
            doc_id = item["id"]
            score = rrf_score(rank, k)
            if doc_id in doc_entries:
                doc_entries[doc_id]["semantic_rank"] = rank
                doc_entries[doc_id]["rrf_score"] += score
            else:
                doc_entries[doc_id] = {
                    "id": doc_id,
                    "title": item["title"],
                    "document": item.get("document") or "",
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": score,
                }

        ranked = sorted(
            doc_entries.values(),
            key=lambda entry: entry["rrf_score"],
            reverse=True,
        )
        return ranked[:limit]