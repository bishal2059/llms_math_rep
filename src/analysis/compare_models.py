from pathlib import Path
import pandas as pd


def compare_model_results(probing_csv: str | Path, clustering_csv: str | Path) -> pd.DataFrame:
    probing = pd.read_csv(probing_csv)
    clustering = pd.read_csv(clustering_csv)

    merged = probing.merge(
        clustering[["model_name", "layer_idx", "silhouette"]],
        on=["model_name", "layer_idx"],
        how="left",
    )
    return merged