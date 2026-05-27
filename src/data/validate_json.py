import json
from pathlib import Path

REQUIRED_KEYS = {"question", "answer_domain"}


def validate_json_dataset(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object.")
        missing = REQUIRED_KEYS - set(row.keys())
        if missing:
            raise ValueError(f"Row {i} missing keys: {missing}")
        if not str(row["question"]).strip():
            raise ValueError(f"Row {i} has empty question.")
        if not str(row["answer_domain"]).strip():
            raise ValueError(f"Row {i} has empty answer_domain.")

    return data