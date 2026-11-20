import pickle
import string
from collections import defaultdict
from typing import Any

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, load_movies, load_stopwords

MAX_RESULTS = 5
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


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
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    return [stemmer.stem(token) for token in filtered_tokens]


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if not tokens:
            return []
        doc_ids = self.index.get(tokens[0], set())
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with self.docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


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