from pathlib import Path
import pandas as pd


def make_final_table(probing_csv: str | Path, clustering_csv: str | Path, out_csv: str | Path):
    probing = pd.read_csv(probing_csv)
    clustering = pd.read_csv(clustering_csv)

    merged = probing.merge(
        clustering[["model_name", "layer_idx", "silhouette_score"]],
        on=["model_name", "layer_idx"],
        how="left",
    )

    summary = (
        merged.groupby("model_name", as_index=False)
        .agg(
            best_accuracy=("accuracy", "max"),
            best_macro_f1=("macro_f1", "max"),
            best_silhouette=("silhouette_score", "max"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_silhouette=("silhouette_score", "mean"),
        )
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary