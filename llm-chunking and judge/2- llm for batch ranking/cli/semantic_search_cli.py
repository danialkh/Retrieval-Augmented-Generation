import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    _load_movies,
    chunk_text,
    embed_chunks,
    embed_query_text,
    embed_text,
    semantic_chunk_text,
    semantic_search,
    verify_embeddings,
    verify_model,
    search_chunks
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

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping words between consecutive chunks (default: 0)",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into semantic sentence-based chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences between consecutive chunks (default: 0)",
    )

    subparsers.add_parser("embed_chunks", help="Build or load chunked movie embeddings")


    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query string")
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

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
    elif args.command == "chunk":
        chunks = chunk_text(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Chunking {len(args.text)} characters")
        for i, chunk in enumerate(chunks, start=1):
            print(f"{i}. {chunk}")
    elif args.command == "semantic_chunk":
        chunks = semantic_chunk_text(
            args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap
        )
        print(f"Semantically chunking {len(args.text)} characters")
        for i, chunk in enumerate(chunks, start=1):
            print(f"{i}. {chunk}")
    elif args.command == "search_chunked":
        documents = _load_movies()
        searcher = ChunkedSemanticSearch()
        searcher.load_or_create_chunk_embeddings(documents)
        results = searcher.search_chunks(args.query, limit=args.limit)

        for i, result in enumerate(results, start=1):
            print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
            print(f"   {result['document']}...")

    elif args.command == "embed_chunks":
        embed_chunks()


if __name__ == "__main__":
    main()