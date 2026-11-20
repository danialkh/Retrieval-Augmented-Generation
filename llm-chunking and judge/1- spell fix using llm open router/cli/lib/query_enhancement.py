import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def enhance_spell(query: str, model: str = DEFAULT_MODEL) -> str:
    client = _get_client()
    prompt = (
        "Fix any spelling errors in the user-provided movie search query below.\n"
        "Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.\n"
        "Preserve punctuation and capitalization unless a change is required for a typo fix.\n"
        "If there are no spelling errors, or if you're unsure, output the original query unchanged.\n"
        "Output only the final query text, nothing else.\n"
        f'User query: "{query}"'
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        return query

    cleaned = content.strip().strip('"').strip("'").strip()
    return cleaned or query


def enhance_query(query: str, method: str) -> str:
    if method == "spell":
        return enhance_spell(query)
    raise ValueError(f"Unsupported enhancement method: {method}")