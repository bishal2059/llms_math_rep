from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired Cohen's d using difference scores.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y

    sd = diff.std(ddof=1)

    if sd == 0:
        return 0.0

    return float(diff.mean() / sd)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:

    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y
    n = len(diff)

    boots = []

    for _ in range(n_boot):

        sample = rng.choice(
            diff,
            size=n,
            replace=True
        )

        boots.append(sample.mean())

    lower = np.quantile(
        boots,
        alpha / 2
    )

    upper = np.quantile(
        boots,
        1 - alpha / 2
    )

    return float(lower), float(upper)


def paired_significance(
    df: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
    layer_col: str = "layer_idx",
    model_col: str = "model_name",
) -> dict:

    a = df[
        df[model_col].str.contains(
            model_a,
            case=False,
            na=False
        )
    ].copy()

    b = df[
        df[model_col].str.contains(
            model_b,
            case=False,
            na=False
        )
    ].copy()

    merged = a[
        [layer_col, metric]
    ].merge(
        b[[layer_col, metric]],
        on=layer_col,
        suffixes=("_a", "_b"),
        how="inner",
    ).sort_values(layer_col)

    x = merged[f"{metric}_a"].to_numpy(dtype=float)
    y = merged[f"{metric}_b"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError(
            "Need at least 2 matched layers."
        )

    try:

        w_stat, w_p = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

    except ValueError:

        w_stat, w_p = np.nan, 1.0

    t_stat, t_p = ttest_rel(x, y)

    d = cohens_d_paired(x, y)

    ci_low, ci_high = bootstrap_ci(
        x,
        y
    )

    return {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_layers": int(len(x)),
        "wilcoxon_stat": (
            float(w_stat)
            if w_stat == w_stat
            else np.nan
        ),
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
        "cohens_d_paired": float(d),
        "boot_ci_low": float(ci_low),
        "boot_ci_high": float(ci_high),
        "mean_diff": float((x - y).mean()),
    }


def save_significance_report(
    report: dict,
    out_path: str | Path
):

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with out_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )