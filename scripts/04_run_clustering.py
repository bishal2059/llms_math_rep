from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.silhouette import (
    compute_silhouette
)

RESULTS = []

if __name__ == "__main__":

    embedding_paths = sorted(
        Path("results/embeddings").rglob(
            "layer_*.npz"
        )
    )

    for npz_path in embedding_paths:

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        embeddings = data["embeddings"]

        score, cluster_ids = compute_silhouette(
            embeddings=embeddings,
            n_clusters=4,
            seed=42,
        )

        row = {
            "model_name": str(
                data["model_name"][0]
            ),
            "layer_idx": int(
                data["layer_idx"][0]
            ),
            "silhouette_score": float(score),
            "file": str(npz_path),
        }

        RESULTS.append(row)

        print(
            f"{row['model_name']} | "
            f"Layer {row['layer_idx']} | "
            f"Silhouette = "
            f"{row['silhouette_score']:.4f}"
        )

    results_df = pd.DataFrame(RESULTS)

    out_path = (
        "results/clustering/"
        "silhouette_results.csv"
    )

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        out_path,
        index=False
    )

    print("Clustering completed.")