from pathlib import Path
import json
import pandas as pd

LABEL_MAP = {
    "Algebra": "algebra",
    "Arithmetic": "arithmetic",
    "Calculus": "calculus",
    "Probability": "probability",
}


def preprocess_json(raw_json_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    raw_json_path = Path(raw_json_path)
    out_path = Path(out_path)

    with raw_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["label"] = df["answer_domain"].map(LABEL_MAP)
    df = df.dropna(subset=["question", "label"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].ne("")]
    df = df.reset_index(drop=True)
    df["sample_id"] = [f"s{i:05d}" for i in range(len(df))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df