import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Movie Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Verify the embedding model setup")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_parser.add_argument("text", type=str, help="Input text to embed")

    subparsers.add_parser("verify_embeddings", help="Build or load and verify movie embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query string")

    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    args = parser.parse_args()

    if args.command == "verify":
        verify_model()
    elif args.command == "embed_text":
        embed_text(args.text)
    elif args.command == "verify_embeddings":
        verify_embeddings()
    elif args.command == "embed_query":
        embed_query_text(args.query)
    elif args.command == "search":
        semantic_search(args.query, limit=args.limit)


if __name__ == "__main__":
    main()