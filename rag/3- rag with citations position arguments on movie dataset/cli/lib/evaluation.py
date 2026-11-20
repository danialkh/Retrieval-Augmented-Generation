import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EVAL_MODEL = "openrouter/free"


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def _format_results(results: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for idx, doc in enumerate(results, start=1):
        title = doc.get("title", "")
        description = doc.get("document", "")
        formatted.append(f"{idx}. {title}: {description}")
    return formatted


def _parse_scores_json(raw_text: str | None, expected_len: int) -> list[int]:
    default_scores = [0] * expected_len
    if not raw_text:
        return default_scores

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", cleaned)
        if not match:
            return default_scores
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default_scores

    if not isinstance(parsed, list):
        return default_scores

    scores: list[int] = []
    for item in parsed[:expected_len]:
        if isinstance(item, bool):
            scores.append(0)
        elif isinstance(item, int) and item in {0, 1, 2, 3}:
            scores.append(item)
        elif isinstance(item, float) and int(item) in {0, 1, 2, 3}:
            scores.append(int(item))
        elif isinstance(item, str) and item.strip() in {"0", "1", "2", "3"}:
            scores.append(int(item.strip()))
        else:
            scores.append(0)

    while len(scores) < expected_len:
        scores.append(0)

    return scores


def evaluate_results(
    query: str,
    results: list[dict[str, Any]],
    model: str = DEFAULT_EVAL_MODEL,
    max_retries: int = 2,
) -> list[tuple[dict[str, Any], int]]:
    if not results:
        return []

    client = _get_client()
    formatted_results = _format_results(results)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
Query: "{query}"
Results:
{chr(10).join(formatted_results)}
Scale:
3: Highly relevant
2: Relevant
1: Marginally relevant
0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.
Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:
[2, 0, 3, 2, 0, 1]"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    scores = [0] * len(results)
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = (
                response.choices[0].message.content if response.choices else None
            )
            parsed = _parse_scores_json(content, len(results))
            if any(s > 0 for s in parsed) or (content and "[" in content):
                scores = parsed
                break
        except Exception:
            if attempt >= max_retries:
                break

    return list(zip(results, scores))