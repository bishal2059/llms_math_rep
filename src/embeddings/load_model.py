from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    return tokenizer, model