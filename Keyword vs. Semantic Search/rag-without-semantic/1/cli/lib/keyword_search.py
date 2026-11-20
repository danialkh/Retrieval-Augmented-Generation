from typing import Any

from .search_utils import load_movies

MAX_RESULTS = 5


def search_movies(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    movies = load_movies()
    results: list[dict[str, Any]] = []

    for movie in movies:
        title = movie.get("title", "")
        if query in title:
            results.append(movie)
            if len(results) >= limit:
                break

    return results