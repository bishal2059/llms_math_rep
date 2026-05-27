from src.data.validate_json import validate_json_dataset

DATASET_PATH = "data/raw/math_dataset.json"

if __name__ == "__main__":
    validate_json_dataset(DATASET_PATH)
    print("Dataset validation successful.")