from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame, label_col: str = "label", seed: int = 42):
    train_parts, test_parts = [], []

    for label, group in df.groupby(label_col):
        if len(group) < 500:
            raise ValueError(f"Class {label} has only {len(group)} rows; expected at least 500.")
        train_df, test_df = train_test_split(
            group,
            train_size=400,
            test_size=100,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        train_parts.append(train_df)
        test_parts.append(test_df)

    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test


def save_split_ids(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df[["sample_id"]].to_csv(out_dir / "train_ids.csv", index=False)
    test_df[["sample_id"]].to_csv(out_dir / "test_ids.csv", index=False)