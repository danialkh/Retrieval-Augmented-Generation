import string
from typing import Any

from .search_utils import load_movies, load_stopwords

MAX_RESULTS = 5
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    return text.lower().translate(PUNCTUATION_TABLE)


def _build_stopwords() -> list[str]:
    processed: list[str] = []
    for word in load_stopwords():
        cleaned = normalize_text(word).strip()
        if cleaned and cleaned not in processed:
            processed.append(cleaned)
    return processed


STOPWORDS: list[str] = _build_stopwords()


def tokenize_text(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    return [token for token in tokens if token not in STOPWORDS]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_movies(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    movies = load_movies()
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    results: list[dict[str, Any]] = []
    for movie in movies:
        title = movie.get("title", "")
        title_tokens = tokenize_text(title)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results