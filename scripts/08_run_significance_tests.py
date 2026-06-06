from pathlib import Path

import pandas as pd

from src.analysis.significance_tests import (
    paired_significance,
    save_significance_report,
)

PROBING_CSV = (
    "results/probing/"
    "layerwise_results.csv"
)

CLUSTERING_CSV = (
    "results/clustering/"
    "silhouette_results.csv"
)

if __name__ == "__main__":

    probing = pd.read_csv(
        PROBING_CSV
    )

    clustering = pd.read_csv(
        CLUSTERING_CSV
    )

    report_acc = paired_significance(
        probing,
        metric="accuracy",
        model_a="qwen",
        model_b="llama",
    )

    report_f1 = paired_significance(
        probing,
        metric="macro_f1",
        model_a="qwen",
        model_b="llama",
    )

    report_sil = paired_significance(
        clustering.rename(
            columns={
                "silhouette_score":
                "silhouette"
            }
        ),
        metric="silhouette",
        model_a="qwen",
        model_b="llama",
    )

    out_dir = Path(
        "results/stats"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_significance_report(
        report_acc,
        out_dir /
        "accuracy_significance.json"
    )

    save_significance_report(
        report_f1,
        out_dir /
        "macro_f1_significance.json"
    )

    save_significance_report(
        report_sil,
        out_dir /
        "silhouette_significance.json"
    )

    print(
        "Significance testing complete."
    )