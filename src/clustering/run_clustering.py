from pathlib import Path
import numpy as np
import pandas as pd

from .silhouette import compute_silhouette


def run_clustering_on_folder(embeddings_dir: str | Path, out_csv: str | Path):
    embeddings_dir = Path(embeddings_dir)
    rows = []

    for npz_path in sorted(embeddings_dir.rglob("layer_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        X = data["embeddings"]
        score, cluster_ids = compute_silhouette(X, n_clusters=4)
        rows.append({
            "file": str(npz_path),
            "model_name": str(data["model_name"][0]),
            "layer_idx": int(data["layer_idx"][0]),
            "silhouette": float(score),
        })

    df = pd.DataFrame(rows).sort_values(["model_name", "layer_idx"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df