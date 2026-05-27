from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_by_layer(csv_path: str | Path, metric: str, out_path: str | Path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    for model_name, group in df.groupby("model_name"):
        group = group.sort_values("layer_idx")
        plt.plot(group["layer_idx"], group[metric], marker="o", label=model_name)
    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()