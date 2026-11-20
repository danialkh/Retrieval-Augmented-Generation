import argparse
import json
from pathlib import Path
from typing import Any

from lib.hybrid_search import HybridSearch
from lib.semantic_search import _load_movies


def _load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for title in top_k if title in relevant)
    return relevant_retrieved / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for title in top_k if title in relevant)
    return relevant_retrieved / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit

    project_root = Path(__file__).resolve().parent.parent
    golden_path = project_root / "data" / "golden_dataset.json"
    test_cases = _load_golden_dataset(golden_path)

    documents = _load_movies()
    hybrid_search = HybridSearch(documents)

    print(f"k={limit}")
    for case in test_cases:
        query = case["query"]
        relevant_docs = list(case.get("relevant_docs", []))
        relevant_set = set(relevant_docs)

        results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [item["title"] for item in results]
        precision = _precision_at_k(retrieved_titles, relevant_set, limit)
        recall = _recall_at_k(retrieved_titles, relevant_set, limit)

        print(f"Query: {query}")
        print(f"  Precision@{limit}: {precision:.4f}")
        print(f"  Recall@{limit}: {recall:.4f}")
        print(f"  Retrieved: {', '.join(retrieved_titles)}")
        print(f"  Relevant: {', '.join(relevant_docs)}")


if __name__ == "__main__":
    main()