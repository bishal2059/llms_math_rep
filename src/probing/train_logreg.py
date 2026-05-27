from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logreg(seed: int = 42, max_iter: int = 2000):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            multi_class="auto",
            n_jobs=None,
        )),
    ])