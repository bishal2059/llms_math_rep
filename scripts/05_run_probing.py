from pathlib import Path

import numpy as np
import pandas as pd

from src.probing.layerwise_probing import (
    run_layerwise_probing
)

RESULTS = []

TRAIN_IDS_PATH = Path("data/splits/train_ids.csv")
TEST_IDS_PATH = Path("data/splits/test_ids.csv")

if __name__ == "__main__":

    if not TRAIN_IDS_PATH.exists() or not TEST_IDS_PATH.exists():
        raise FileNotFoundError(
            "Missing split files. Run scripts/02_split_dataset.py first."
        )

    train_ids = set(
        pd.read_csv(TRAIN_IDS_PATH)["sample_id"].astype(str)
    )
    test_ids = set(
        pd.read_csv(TEST_IDS_PATH)["sample_id"].astype(str)
    )

    embedding_paths = sorted(
        Path("results/embeddings").rglob(
            "layer_*.npz"
        )
    )

    for npz_path in embedding_paths:

        result = run_layerwise_probing(
            npz_path=npz_path,
            train_ids=train_ids,
            test_ids=test_ids,
            seed=42,
        )

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        row = {
            "model_name": str(
                data["model_name"][0]
            ),
            "layer_idx": int(
                data["layer_idx"][0]
            ),
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "file": str(npz_path),
        }

        RESULTS.append(row)

        print(
            f"{row['model_name']} | "
            f"Layer {row['layer_idx']} | "
            f"Accuracy = "
            f"{row['accuracy']:.4f}"
        )

    results_df = pd.DataFrame(RESULTS)

    out_path = (
        "results/probing/"
        "layerwise_results.csv"
    )

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        out_path,
        index=False
    )

    print("Layer-wise probing complete.")