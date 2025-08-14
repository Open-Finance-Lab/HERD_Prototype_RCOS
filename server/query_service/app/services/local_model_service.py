from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cache for loaded models and tokenizers
_model_cache = {}
_tokenizer_cache = {}

def load_local_model(model_path: str):
    """Load a Hugging Face model from a local path, cached after first load."""
    if model_path not in _model_cache:
        print(f"[INFO] Loading model from {model_path}...")
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
        _model_cache[model_path] = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    return _model_cache[model_path], _tokenizer_cache[model_path]

def run_local_model(model_path: str, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate text from a local model."""
    model, tokenizer = load_local_model(model_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
