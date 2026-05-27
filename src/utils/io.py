from pathlib import Path
import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)