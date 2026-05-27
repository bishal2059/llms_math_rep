from src.data.preprocess_json import preprocess_json

RAW_PATH = "data/raw/math_dataset.json"
OUT_PATH = "data/processed/dataset.parquet"

if __name__ == "__main__":
    preprocess_json(RAW_PATH, OUT_PATH)
    print("Dataset preprocessing completed.")