import math
import os
import pickle
import re
import string
from collections import Counter, defaultdict
from typing import Any

from nltk.stem import PorterStemmer

CACHE_DIR = "cache"
STOPWORDS_PATH = os.path.join("data", "stopwords.txt")
BM25_K1 = 1.5
BM25_B = 0.75


def _load_stopwords() -> set[str]:
    if not os.path.exists(STOPWORDS_PATH):
        return set()
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    stopwords = _load_stopwords()

    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    raw_tokens = [tok for tok in re.split(r"\s+", no_punct) if tok]

    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in stopwords:
            continue
        stemmed = stemmer.stem(tok)
        if stemmed and stemmed not in stopwords:
            tokens.append(stemmed)
    return tokens


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single token after preprocessing, got {tokens}")
    return tokens[0]


class InvertedIndex:
    def __init__(self, cache_dir: str = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.index_path = os.path.join(cache_dir, "index.pkl")
        self.docmap_path = os.path.join(cache_dir, "docmap.pkl")
        self.tf_path = os.path.join(cache_dir, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(cache_dir, "doc_lengths.pkl")
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

        
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize_text(term)
        if not tokens:
            return []
        matching_ids: set[int] = set()
        for token in tokens:
            matching_ids |= self.index.get(token, set())
        return sorted(matching_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if not tokens:
            return 0
        if len(tokens) != 1:
            raise ValueError(f"Expected a single token after preprocessing, got {tokens}")
        return self.term_frequencies.get(doc_id, Counter()).get(tokens[0], 0)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0

        avg_doc_len = self.__get_avg_doc_length()
        if avg_doc_len == 0.0:
            return float(tf)

        doc_len = self.doc_lengths.get(doc_id, 0)
        length_norm = 1.0 - b + b * (doc_len / avg_doc_len)
        return (tf * (k1 + 1.0)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if not tokens:
            return 0.0
        if len(tokens) != 1:
            raise ValueError(f"Expected a single token after preprocessing, got {tokens}")

        token = tokens[0]
        doc_count = len(self.docmap)
        if doc_count == 0:
            return 0.0

        df = len(self.index.get(token, set()))
        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def bm25(self, doc_id: int, term: str) -> float:
        if not tokenize_text(term):
            return 0.0
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 10) -> list[tuple[dict[str, Any], float]]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        candidate_ids: set[int] = set()
        for token in query_tokens:
            candidate_ids |= self.index.get(token, set())

        scores: dict[int, float] = {}
        for doc_id in candidate_ids:
            total_score = 0.0
            for token in query_tokens:
                if not tokenize_text(token):
                    continue
                total_score += self.bm25(doc_id, token)
            if total_score > 0.0:
                scores[doc_id] = total_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            (self.docmap[doc_id], score)
            for doc_id, score in ranked
            if doc_id in self.docmap
        ]

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def build(self, documents: list[dict[str, Any]]) -> None:
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()

        for doc in documents:
            doc_id = doc["id"]
            self.docmap[doc_id] = doc
            text = f"{doc.get('title', '')} {doc.get('description', '')}"
            self.__add_document(doc_id, text)

    def save(self, cache_dir: str = CACHE_DIR) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(cache_dir, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(os.path.join(cache_dir, "doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self, cache_dir: str = CACHE_DIR) -> None:
        with open(os.path.join(cache_dir, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(cache_dir, "docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)
        with open(os.path.join(cache_dir, "term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(os.path.join(cache_dir, "doc_lengths.pkl"), "rb") as f:
            self.doc_lengths = pickle.load(f)