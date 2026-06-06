from pathlib import Path

from src.analysis.visualize_representations import (
    plot_layer_embedding_2d
)

EMBED_DIR = Path(
    "results/embeddings"
)

OUT_DIR = Path(
    "results/figures/pca"
)

if __name__ == "__main__":

    npz_files = sorted(
        EMBED_DIR.rglob(
            "layer_*.npz"
        )
    )

    for npz_path in npz_files:

        model_name = (
            npz_path.parent.name
        )

        layer_name = (
            npz_path.stem
        )

        out_path = (
            OUT_DIR /
            model_name /
            f"{layer_name}_pca.png"
        )

        info = plot_layer_embedding_2d(
            npz_path=npz_path,
            out_path=out_path,
            method="pca",
            max_points=1500,
            seed=42,
        )

        print(
            f"Saved: "
            f"{info['output_file']}"
        )