import requests, json, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding.embedder import Embedder

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3.1:8b-instruct")
TEMP         = float(os.getenv("LLM_TEMPERATURE", "0.1"))
NUM_PREDICT  = int(os.getenv("LLM_NUM_PREDICT", "400"))
TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "60"))

SYSTEM_SPECIALIZER = (
    "You are a strict prompt editor. "
    "Specialize the GENERAL prompt to the requested TOPIC only. "
    "Keep the original task and constraints that are relevant to the topic. "
    "Remove or reword anything out-of-scope. "
    "Return ONLY the final specialized prompt text—no JSON, no explanations."
)

def call_ollama_specialize(topic: str, general_prompt: str) -> str:
    """Call Ollama /api/chat to get a specialized prompt for a given topic."""
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
            {
                "role": "user",
                "content": (
                    f"TOPIC: {topic}\n\n"
                    "GENERAL PROMPT:\n"
                    f"{general_prompt}\n\n"
                    "Instructions:\n"
                    "- Rewrite the prompt so it focuses strictly on the TOPIC above.\n"
                    "- Keep the original intent and constraints that are relevant to this topic.\n"
                    "- If critical info is missing, add concise questions asking for it.\n"
                    "- Use concise wording. Output only the specialized prompt."
                ),
            },
        ],
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama error for topic '{topic}': {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Ollama response parse error for '{topic}': {e}")

embedder = Embedder("topic_vector.json", model_name="all-MiniLM-L6-v2")

app = FastAPI()

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3

@app.post("/classify")
def classify(req: QueryRequest):
    results = embedder(req.text, top_k=req.top_k)
    return {"topics": results}

@app.post("/create_prompts")
def create_prompts(req: QueryRequest):
    topics = embedder(req.text, top_k=req.top_k) 
    if not topics:
        return {"original": req.text, "topics": [], "prompts": {}}

    prompts = {}
    for t in topics:
        topic_name = t["topic"]
        score = float(t.get("score", 0.0))
        specialized = call_ollama_specialize(topic_name, req.text)
        prompts[topic_name] = {
            "score": score,
            "prompt": specialized,
        }

    return {
        "original": req.text,
        "model": MODEL_NAME,
        "topics": topics,
        "prompts": prompts
    }


