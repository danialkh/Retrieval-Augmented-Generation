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


def summarize_results(
    query: str,
    results: list[dict[str, Any]],
    model: str = DEFAULT_RAG_MODEL,
) -> str:
    client = _get_client()
    formatted_results = _format_docs(results)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Webflyx users. Webflyx is a movie streaming service.
Query: {query}
Search results:
{formatted_results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

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

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to retrieve and summarize (default: 5)",
    )

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
        case "summarize":
            query = args.query
            limit = args.limit
            documents = _load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=limit)

            print("Search Results:")
            for item in results:
                print(f"  - {item['title']}")

            summary = summarize_results(query, results)
            print("LLM Summary:")
            print(summary)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()