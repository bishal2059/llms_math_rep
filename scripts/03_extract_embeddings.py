from pathlib import Path

import pandas as pd

from src.embeddings.load_model import (
    load_model_and_tokenizer
)

from src.embeddings.extract_layers import (
    extract_all_layers
)

from src.embeddings.save_npz import (
    save_layer_npz
)

from src.utils.config import (
    load_yaml
)

CONFIG_PATH = "configs/models.yaml"

DATA_PATH = "data/processed/dataset.parquet"

if __name__ == "__main__":

    config = load_yaml(CONFIG_PATH)
    models_config = config.get("models", {})
    device_preference = config.get("device", "cuda_if_available")

    if not models_config:
        raise ValueError("No models found in configs/models.yaml")

    df = pd.read_parquet(DATA_PATH)

    texts = df["question"].tolist()
    labels = df["label"].to_numpy()
    sample_ids = df["sample_id"].tolist()

    for model_tag, model_cfg in models_config.items():

        model_name = model_cfg["model_name"]

        print(f"Loading model: {model_name}")

        tokenizer, model, device = load_model_and_tokenizer(
            model_name,
            device_preference=device_preference,
        )

        print(f"Resolved device: {device}")

        layer_embeddings = extract_all_layers(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=8,
            max_length=256,
        )

        for layer_idx, embeddings in enumerate(
            layer_embeddings
        ):

            out_path = (
                f"results/embeddings/"
                f"{model_tag}/"
                f"layer_{layer_idx:02d}.npz"
            )

            save_layer_npz(
                out_path=out_path,
                embeddings=embeddings,
                labels=labels,
                questions=texts,
                sample_ids=sample_ids,
                model_name=model_name,
                layer_idx=layer_idx,
            )

            print(
                f"Saved layer {layer_idx} "
                f"for {model_tag}"
            )

    print("Embedding extraction complete.")