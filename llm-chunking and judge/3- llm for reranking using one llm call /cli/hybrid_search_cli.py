import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.query_enhancement import enhance_query
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

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion hybrid search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k constant (default: 60)",
    )
    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to display (default: 5)",
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch"],
        help="Re-ranking method",
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
        case "rrf-search":
            search_query = args.query
            if args.enhance:
                enhanced_query = enhance_query(args.query, args.enhance)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                search_query = enhanced_query

            if args.rerank_method:
                print(
                    f"Re-ranking top {args.limit} results using {args.rerank_method} method..."
                )

            documents = _load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(
                search_query,
                k=args.k,
                limit=args.limit,
                rerank_method=args.rerank_method,
            )

            print(f"Reciprocal Rank Fusion Results for '{search_query}' (k={args.k}):")
            for rank, item in enumerate(results, start=1):
                print(f"{rank}. {item['title']}")
                if "rerank_rank" in item:
                    rerank_part = f"   Re-rank Rank: {item['rerank_rank']}"
                elif "rerank_score" in item:
                    rerank_part = f"   Re-rank Score: {item['rerank_score']:.3f}/10"
                else:
                    rerank_part = ""
                print(
                    f"  {rerank_part}   RRF Score: {item['rrf_score']:.3f}   "
                    f"BM25 Rank: {item['bm25_rank']}, Semantic Rank: {item['semantic_rank']}   "
                    f"{item['document'][:100]}..."
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()