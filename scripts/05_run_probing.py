from pathlib import Path

import numpy as np
import pandas as pd

from src.probing.layerwise_probing import (
    run_layerwise_probing
)

RESULTS = []

if __name__ == "__main__":

    embedding_paths = sorted(
        Path("results/embeddings").rglob(
            "layer_*.npz"
        )
    )

    for npz_path in embedding_paths:

        result = run_layerwise_probing(
            npz_path=npz_path,
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