from transformers import AutoTokenizer, AutoModel


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=True
    )

    return tokenizer, model