from pathlib import Path
import pandas as pd


def summarise_clustering_results(results_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    score_col = (
        "silhouette_score"
        if "silhouette_score" in df.columns
        else "silhouette"
    )

    summary = (
        df.groupby("model_name", as_index=False)
        .agg(
            best_layer=(
                "layer_idx",
                lambda s: int(df.loc[s.index, score_col].idxmax()),
            ),
            mean_silhouette=(score_col, "mean"),
            max_silhouette=(score_col, "max"),
        )
    )
    return summary