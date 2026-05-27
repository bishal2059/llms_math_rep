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

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

DATA_PATH = "data/processed/dataset.parquet"

if __name__ == "__main__":

    df = pd.read_parquet(DATA_PATH)

    texts = df["question"].tolist()
    labels = df["label"].to_numpy()
    sample_ids = df["sample_id"].tolist()

    for model_tag, model_name in MODELS.items():

        print(f"Loading model: {model_name}")

        tokenizer, model = load_model_and_tokenizer(
            model_name
        )

        layer_embeddings = extract_all_layers(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=model.device,
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