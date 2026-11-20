import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MOVIES_PATH = DATA_DIR / "movies.json"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"


def load_movies_data(filepath: Path = MOVIES_PATH) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_movies(filepath: Path = MOVIES_PATH) -> list[dict[str, Any]]:
    data = load_movies_data(filepath)
    return data.get("movies", [])


def load_stopwords(filepath: Path = STOPWORDS_PATH) -> list[str]:
    with filepath.open("r", encoding="utf-8") as f:
        return f.read().splitlines()