# app/core/settings.py
import os, json
from pathlib import Path

def load_topic_to_model(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    for topic, path in data.items():
        if isinstance(path, str):
            data[topic] = os.path.expandvars(path)
    return data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPERT_STORE = os.getenv("EXPERT_STORE", "experts.json")
expert_path = DATA_DIR / EXPERT_STORE

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma:2b")
TEMP = float(os.getenv("LLM_TEMPERATURE", "0.1"))
NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "400"))
TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "60"))
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1/completions")

SYSTEM_SPECIALIZER = (
    "You are a strict prompt editor. Specialize the GENERAL prompt to the requested TOPIC only. "
    "Keep the original task and constraints that are relevant to the topic. "
    "Remove or reword anything out-of-scope. Return ONLY the final specialized prompt text—no JSON, no explanations."
)

TOPIC_TO_MODEL = load_topic_to_model(expert_path)
print(TOPIC_TO_MODEL)