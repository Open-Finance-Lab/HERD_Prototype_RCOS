from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    model_name = "gpt2"  # Using GPT-2 as the base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer