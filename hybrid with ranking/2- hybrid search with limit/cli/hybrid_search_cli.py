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
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Keyword weight vs semantic weight (default: 0.5)",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to display (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = _load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(
                args.query, alpha=args.alpha, limit=args.limit
            )
            for rank, item in enumerate(results, start=1):
                print(f"{rank}. {item['title']}")
                print(
                    f"  Hybrid Score: {item['hybrid_score']:.3f}  "
                    f"BM25: {item['bm25_score']:.3f}, Semantic: {item['semantic_score']:.3f}  "
                    f"{item['document'][:100]}..."
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()