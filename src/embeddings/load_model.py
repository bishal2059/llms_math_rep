import torch
from transformers import AutoModel, AutoTokenizer


def resolve_device(device_preference: str | None) -> torch.device:
    preference = (device_preference or "cuda_if_available").lower()

    if preference in {"cuda_if_available", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preference == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    device_preference: str | None = "cuda_if_available",
):
    device = resolve_device(device_preference)
    device_map = "auto" if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    return tokenizer, model, device