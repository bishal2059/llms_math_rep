from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from .pooling import mean_pool


def _normalize_device_map_entry(mapped) -> torch.device | None:
    """Convert accelerate device_map values to torch.device."""
    if isinstance(mapped, torch.device):
        return mapped

    if isinstance(mapped, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{mapped}")
        return torch.device("cpu")

    mapped_str = str(mapped).strip().lower()
    if mapped_str in {"disk", "meta"}:
        return None
    if mapped_str == "cpu":
        return torch.device("cpu")
    if mapped_str.startswith("cuda"):
        return torch.device(mapped_str)
    if mapped_str.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{mapped_str}")
        return torch.device("cpu")

    return None


def _select_input_device(model, fallback_device: str | torch.device) -> torch.device:
    if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map"):
        for mapped in model.hf_device_map.values():
            device = _normalize_device_map_entry(mapped)
            if device is not None and device.type == "cuda":
                return device

        for mapped in model.hf_device_map.values():
            device = _normalize_device_map_entry(mapped)
            if device is not None:
                return device

    return torch.device(fallback_device)


def extract_all_layers(
    model,
    tokenizer,
    texts: list[str],
    device: str | torch.device,
    batch_size: int = 8,
    max_length: int = 256,
    include_embedding_layer: bool = True,
):
    model.eval()

    if not (hasattr(model, "hf_device_map") and getattr(model, "hf_device_map")):
        model.to(device)

    input_device = _select_input_device(model, device)

    all_layers = None

    for start in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        layer_range = range(len(hidden_states)) if include_embedding_layer else range(1, len(hidden_states))
        batch_by_layer = []
        for layer_idx in layer_range:
            pooled = mean_pool(hidden_states[layer_idx], inputs["attention_mask"])
            batch_by_layer.append(pooled.detach().cpu().numpy())

        if all_layers is None:
            all_layers = [[] for _ in range(len(batch_by_layer))]

        for i, arr in enumerate(batch_by_layer):
            all_layers[i].append(arr)

    all_layers = [np.concatenate(chunks, axis=0) for chunks in all_layers]
    return all_layers