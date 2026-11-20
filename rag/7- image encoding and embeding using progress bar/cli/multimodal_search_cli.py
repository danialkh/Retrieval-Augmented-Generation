import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify-image-embedding",
        help="Verify image embedding generation and print embedding dimensionality",
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file",
    )

    image_search_parser = subparsers.add_parser(
        "image-search",
        help="Search movies using an input image",
    )
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the query image file",
    )

    args = parser.parse_args()

    match args.command:
        case "verify-image-embedding":
            verify_image_embedding(args.image_path)
        case "image-search":
            results = image_search_command(args.image_path)
            for rank, item in enumerate(results, start=1):
                snippet = item["description"][:100]
                print(
                    f"{rank}. {item['title']} (similarity: {item['similarity']:.3f})\n"
                    f"   {snippet}..."
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()