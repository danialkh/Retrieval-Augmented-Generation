import json
from pathlib import Path
from typing import Any

MOVIES_PATH = Path(__file__).resolve().parents[2] / "data" / "movies.json"


def load_movies_data(filepath: Path = MOVIES_PATH) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_movies(filepath: Path = MOVIES_PATH) -> list[dict[str, Any]]:
    data = load_movies_data(filepath)
    return data.get("movies", [])