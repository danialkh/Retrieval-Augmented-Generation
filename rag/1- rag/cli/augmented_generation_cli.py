import argparse
import os
from typing import Any

from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from lib.semantic_search import _load_movies
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RAG_MODEL = "openrouter/free"


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def _format_docs(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(results, start=1):
        title = doc.get("title", "")
        description = doc.get("document", "")
        lines.append(f"{idx}. {title}: {description}")
    return "\n".join(lines)


def generate_rag_response(
    query: str,
    results: list[dict[str, Any]],
    model: str = DEFAULT_RAG_MODEL,
) -> str:
    client = _get_client()
    docs = _format_docs(results)
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.
Query: {query}
Documents:
{docs}
Answer:"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content if response.choices else None
    return (content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = _load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=5)

            print("Search Results:")
            for item in results:
                print(f"  - {item['title']}")

            answer = generate_rag_response(query, results)
            print("RAG Response:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()