from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

try:

    import umap

    UMAP_AVAILABLE = True

except Exception:

    UMAP_AVAILABLE = False


def reduce_to_2d(
    X: np.ndarray,
    method: str = "umap",
    seed: int = 42,
):

    method = method.lower()

    if method == "umap" and UMAP_AVAILABLE:

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )

        return reducer.fit_transform(X)

    reducer = PCA(
        n_components=2,
        random_state=seed
    )

    return reducer.fit_transform(X)


def plot_layer_embedding_2d(
    npz_path: str | Path,
    out_path: str | Path,
    method: str = "umap",
    max_points: int = 1500,
    seed: int = 42,
):

    npz_path = Path(npz_path)
    out_path = Path(out_path)

    data = np.load(
        npz_path,
        allow_pickle=True
    )

    X = data["embeddings"]

    y = data["labels"].astype(str)

    if len(X) > max_points:

        rng = np.random.default_rng(seed)

        idx = rng.choice(
            len(X),
            size=max_points,
            replace=False
        )

        X = X[idx]
        y = y[idx]

    X2 = reduce_to_2d(
        X,
        method=method,
        seed=seed
    )

    labels = sorted(set(y.tolist()))

    label_to_id = {
        lab: i
        for i, lab in enumerate(labels)
    }

    c = np.array([
        label_to_id[v]
        for v in y
    ])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X2[:, 0],
        X2[:, 1],
        c=c,
        s=10,
        alpha=0.75
    )

    plt.title(
        f"{method.upper()} projection: "
        f"{npz_path.stem}"
    )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        out_path,
        dpi=300
    )

    plt.close()

    return {
        "input_file": str(npz_path),
        "output_file": str(out_path),
        "method": method,
        "n_points": int(len(X2)),
        "labels": labels,
        "umap_available": UMAP_AVAILABLE,
    }