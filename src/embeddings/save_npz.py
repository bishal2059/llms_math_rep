from pathlib import Path
import json
import numpy as np


def save_layer_npz(
    out_path: str | Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    questions: list[str],
    sample_ids: list[str],
    model_name: str,
    layer_idx: int,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=labels,
        questions=np.array(questions, dtype=object),
        sample_ids=np.array(sample_ids, dtype=object),
        model_name=np.array([model_name], dtype=object),
        layer_idx=np.array([layer_idx], dtype=np.int32),
    )