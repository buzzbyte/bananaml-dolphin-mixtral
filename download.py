from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model() -> tuple:
    """Download the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained("TheBloke/dolphin-2.7-mixtral-8x7b-AWQ")
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/dolphin-2.7-mixtral-8x7b-AWQ")
    return model, tokenizer

if __name__ == "__main__":
    download_model()
