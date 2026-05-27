import pandas as pd

from src.data.split_data import (
    make_split,
    save_split_ids
)

DATA_PATH = "data/processed/dataset.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)

    train_df, test_df = make_split(
        df,
        label_col="label",
        seed=42
    )

    save_split_ids(
        train_df,
        test_df,
        "data/splits"
    )

    print("Train/test split saved.")