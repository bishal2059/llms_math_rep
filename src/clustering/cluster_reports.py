from pathlib import Path
import pandas as pd


def summarise_clustering_results(results_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    summary = (
        df.groupby("model_name", as_index=False)
        .agg(
            best_layer=("layer_idx", lambda s: int(df.loc[s.index, "silhouette"].idxmax())),
            mean_silhouette=("silhouette", "mean"),
            max_silhouette=("silhouette", "max"),
        )
    )
    return summary