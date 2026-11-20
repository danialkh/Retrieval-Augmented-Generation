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


def _call_llm(prompt: str, fallback: str, model: str = DEFAULT_MODEL) -> str:
    client = _get_client()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        return fallback

    cleaned = content.strip().strip('"').strip("'").strip()
    return cleaned or fallback


def enhance_spell(query: str, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "Fix any spelling errors in the user-provided movie search query below.\n"
        "Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.\n"
        "Preserve punctuation and capitalization unless a change is required for a typo fix.\n"
        "If there are no spelling errors, or if you're unsure, output the original query unchanged.\n"
        "Output only the final query text, nothing else.\n"
        f'User query: "{query}"'
    )
    return _call_llm(prompt, fallback=query, model=model)


def enhance_rewrite(query: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return _call_llm(prompt, fallback=query, model=model)


def enhance_expand(query: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    extra_terms = _call_llm(prompt, fallback="", model=model)
    if not extra_terms:
        return query
    if extra_terms.lower().startswith(query.lower()):
        return extra_terms
    return f"{query} {extra_terms}".strip()


def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return enhance_spell(query)
        case "rewrite":
            return enhance_rewrite(query)
        case "expand":
            return enhance_expand(query)
        case _:
            raise ValueError(f"Unsupported enhancement method: {method}")