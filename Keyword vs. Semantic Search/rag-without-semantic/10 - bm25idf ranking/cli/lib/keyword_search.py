import math
import pickle
import string
from collections import Counter, defaultdict
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


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single token after preprocessing, got {tokens}")
    return tokens[0]


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.term_frequencies_path = CACHE_DIR / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if not tokens:
            return []
        doc_ids = self.index.get(tokens[0], set())
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_term(term)
        doc_counts = self.term_frequencies.get(doc_id)
        if not doc_counts:
            return 0
        return doc_counts.get(token, 0)

    def get_idf(self, term: str) -> float:
        token = tokenize_term(term)
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_term(term)
        n = len(self.docmap)
        df = len(self.index.get(token, set()))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

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
        with self.term_frequencies_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if (
            not self.index_path.exists()
            or not self.docmap_path.exists()
            or not self.term_frequencies_path.exists()
        ):
            raise FileNotFoundError("Index cache files not found. Run 'build' first.")
        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)
        with self.docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)
        with self.term_frequencies_path.open("rb") as f:
            self.term_frequencies = pickle.load(f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_term(term)
    return idx.get_tf(doc_id, token)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_term(term)
    return idx.get_bm25_idf(token)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)
    return tf * idf


def search_movies(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    seen_ids: set[int] = set()
    results: list[dict[str, Any]] = []

    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(idx.docmap[doc_id])
                if len(results) >= limit:
                    return results

    return results