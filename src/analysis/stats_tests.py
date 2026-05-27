from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon


def compare_models_wilcoxon(csv_path: str | Path, metric: str = "accuracy"):
    df = pd.read_csv(csv_path)
    qwen = df[df["model_name"].str.contains("qwen", case=False)][metric].values
    llama = df[df["model_name"].str.contains("llama", case=False)][metric].values

    n = min(len(qwen), len(llama))
    stat, p = wilcoxon(qwen[:n], llama[:n])
    return {"statistic": float(stat), "p_value": float(p)}