import torch
import numpy as np
from tqdm import tqdm

from src.embeddings.embedding_utils import mean_pool


@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, device="cuda"):
    model.to(device)
    model.eval()

    all_layer_embeddings = {}

    for text in tqdm(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        hidden_states = outputs.hidden_states

        for layer_idx, layer_hidden in enumerate(hidden_states):
            pooled = mean_pool(layer_hidden, inputs["attention_mask"])
            pooled = pooled.squeeze(0).cpu().numpy()

            if layer_idx not in all_layer_embeddings:
                all_layer_embeddings[layer_idx] = []

            all_layer_embeddings[layer_idx].append(pooled)

    for layer_idx in all_layer_embeddings:
        all_layer_embeddings[layer_idx] = np.array(
            all_layer_embeddings[layer_idx]
        )

    return all_layer_embeddings