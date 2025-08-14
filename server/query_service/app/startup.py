import requests
from app.core.settings import VLLM_URL, TOPIC_TO_MODEL

def preload_all_models():
    for topic, model_id in TOPIC_TO_MODEL.items():
        try:
            print(f"Preloading model '{model_id}' for topic '{topic}'...")
            r = requests.post(
                VLLM_URL.replace("/v1/completions", "/internal/load"),
                json={"model_id": model_id},
                timeout=60
            )
            r.raise_for_status()
            print(f"Loaded model: {model_id}")
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")