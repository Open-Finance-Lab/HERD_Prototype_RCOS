import requests
from fastapi import HTTPException
from app.core.settings import OLLAMA_URL, MODEL_NAME, TEMP, NUM_PREDICT, TIMEOUT_SECS, SYSTEM_SPECIALIZER

def call_ollama_specialize(topic: str, general_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "options": {
            "temperature": TEMP,
            "num_predict": NUM_PREDICT,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.05,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_SPECIALIZER},
            {"role": "user", "content": f"TOPIC: {topic}\n\nGENERAL PROMPT:\n{general_prompt}\n\nInstructions:\n- Rewrite..."}
        ],
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error for '{topic}': {e}")
