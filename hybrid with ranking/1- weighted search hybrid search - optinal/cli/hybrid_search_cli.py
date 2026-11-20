import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.semantic_search import _load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search (BM25 + semantic)"
    )
    weighted_parser.add_argument("query", type=str, help="Search query string")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for semantic search score vs BM25 (default: 0.5)",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = _load_movies()
            hybrid = HybridSearch(documents)
            results = hybrid.weighted_search(
                args.query, alpha=args.alpha, limit=args.limit
            )
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
                print(
                    f"  Hybrid Score: {res['hybrid_score']:.3f}  "
                    f"BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}  "
                    f"{res['document'][:100]}..."
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()