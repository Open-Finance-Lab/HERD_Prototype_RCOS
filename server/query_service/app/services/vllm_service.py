import requests
from app.core.settings import VLLM_URL, TEMP, TIMEOUT_SECS

def call_vllm(model_id: str, prompt: str) -> str:
    payload = {"model": model_id, "prompt": prompt, "max_tokens": 256, "temperature": TEMP}
    try:
        r = requests.post(VLLM_URL, json=payload, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        return r.json().get("choices", [{}])[0].get("text", "[ERROR] No output")
    except Exception as e:
        return f"[ERROR] VLLM error: {e}"