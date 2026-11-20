import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
DEFAULT_SLEEP_SECONDS = 3.0


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def _build_individual_prompt(query: str, doc: dict[str, Any]) -> str:
    title = doc.get("title", "")
    description = doc.get("document", "")
    return f"""Rate how well this movie matches the search query.
Query: "{query}"
Movie: {title}
{description}
Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness
Rate 0-10 (10 = perfect match).Output ONLY the number in your response, no other text or explanation.
Score:"""


def _build_batch_prompt(query: str, documents: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for doc in documents:
        doc_id = doc.get("id")
        title = doc.get("title", "")
        description = doc.get("document", "")
        lines.append(f"ID: {doc_id} | Title: {title}\n{description}")
    doc_list_str = "\n\n".join(lines)

    return f"""Rank the movies listed below by relevance to the following search query.
Query: "{query}"
Movies:
{doc_list_str}
Return the movie IDs in order of relevance, best match first.
Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.
For example:[75, 12, 34, 2, 1]
Ranking:"""


def _parse_score(raw_text: str | None) -> float:
    if not raw_text:
        return 0.0
    match = re.search(r"(?:10(?:\.0+)?|[0-9](?:\.[0-9]+)?)", raw_text.strip())
    if not match:
        return 0.0
    value = float(match.group(0))
    return max(0.0, min(10.0, value))


def _parse_ranked_ids(raw_text: str | None) -> list[int | str]:
    if not raw_text:
        return []
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", cleaned)
        if not match:
            return []
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        return []

    ranked_ids: list[int | str] = []
    for item in parsed:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            ranked_ids.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            ranked_ids.append(int(item.strip()))
    return ranked_ids


def _score_single_document(
    client: OpenAI,
    query: str,
    doc: dict[str, Any],
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> float:
    prompt = _build_individual_prompt(query, doc)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = response.choices[0].message.content if response.choices else None
            parsed = _parse_score(content)
            if parsed > 0.0 or (content and content.strip().startswith("0")):
                return parsed
            if attempt < max_retries:
                time.sleep(1.5)
        except Exception:
            if attempt >= max_retries:
                return 0.0
            time.sleep(2.0 * (attempt + 1))

    return 0.0


def rerank_individual(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 5,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    client = _get_client()
    scored_docs: list[dict[str, Any]] = []

    for idx, doc in enumerate(documents):
        if idx > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        rerank_score = _score_single_document(
            client=client,
            query=query,
            doc=doc,
            model=model,
        )
        updated = dict(doc)
        updated["rerank_score"] = rerank_score
        scored_docs.append(updated)

    ranked = sorted(
        scored_docs,
        key=lambda entry: (entry["rerank_score"], entry.get("rrf_score", 0.0)),
        reverse=True,
    )
    return ranked[:limit]


def rerank_batch(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 5,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    client = _get_client()
    prompt = _build_batch_prompt(query, documents)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    ranked_ids: list[int | str] = []
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = response.choices[0].message.content if response.choices else None
            ranked_ids = _parse_ranked_ids(content)
            if ranked_ids:
                break
        except Exception:
            if attempt >= max_retries:
                break
        if attempt < max_retries:
            time.sleep(2.0 * (attempt + 1))

    doc_by_id: dict[int | str, dict[str, Any]] = {
        doc["id"]: dict(doc) for doc in documents if "id" in doc
    }
    ordered_docs: list[dict[str, Any]] = []
    seen_ids: set[int | str] = set()

    for doc_id in ranked_ids:
        if doc_id in doc_by_id and doc_id not in seen_ids:
            ordered_docs.append(doc_by_id[doc_id])
            seen_ids.add(doc_id)

    for doc in documents:
        doc_id = doc.get("id")
        if doc_id not in seen_ids:
            ordered_docs.append(dict(doc))
            if doc_id is not None:
                seen_ids.add(doc_id)

    reranked: list[dict[str, Any]] = []
    for rank_idx, doc in enumerate(ordered_docs, start=1):
        doc["rerank_rank"] = rank_idx
        reranked.append(doc)

    reranked.sort(key=lambda entry: entry["rerank_rank"])
    return reranked[:limit]