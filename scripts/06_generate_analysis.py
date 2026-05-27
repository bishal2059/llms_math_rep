from src.analysis.make_tables import (
    make_final_table
)

from src.analysis.make_figures import (
    plot_metric_by_layer
)

if __name__ == "__main__":

    probing_csv = (
        "results/probing/"
        "layerwise_results.csv"
    )

    clustering_csv = (
        "results/clustering/"
        "silhouette_results.csv"
    )

    make_final_table(
        probing_csv=probing_csv,
        clustering_csv=clustering_csv,
        out_csv=(
            "results/tables/"
            "final_summary.csv"
        ),
    )

    plot_metric_by_layer(
        csv_path=probing_csv,
        metric="accuracy",
        out_path=(
            "results/figures/"
            "accuracy_by_layer.png"
        ),
    )

    plot_metric_by_layer(
        csv_path=clustering_csv,
        metric="silhouette_score",
        out_path=(
            "results/figures/"
            "silhouette_by_layer.png"
        ),
    )

    print("Final analysis generated.")