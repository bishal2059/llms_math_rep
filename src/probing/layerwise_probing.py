from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .train_logreg import build_logreg
from .evaluate_logreg import evaluate


def run_layerwise_probing(npz_path: str | Path, train_ids: set[str] | None = None, test_ids: set[str] | None = None, seed: int = 42):
    data = np.load(npz_path, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]
    sample_ids = data["sample_ids"].astype(str)

    if train_ids is not None and test_ids is not None:
        train_mask = np.array([sid in train_ids for sid in sample_ids])
        test_mask = np.array([sid in test_ids for sid in sample_ids])
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

    model = build_logreg(seed=seed)
    model.fit(X_train, y_train)
    return evaluate(model, X_test, y_test)