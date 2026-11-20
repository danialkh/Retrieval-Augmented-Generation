import argparse
import sys

from lib.keyword_search import build_command, search_movies, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                results = search_movies(args.query)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            for idx, movie in enumerate(results, start=1):
                print(f"{idx}. ({movie['id']}) {movie['title']}")
        case "build":
            build_command()
        case "tf":
            try:
                tf = tf_command(args.doc_id, args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(tf)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()