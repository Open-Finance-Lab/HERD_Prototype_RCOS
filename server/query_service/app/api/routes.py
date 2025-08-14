# app/api/routes.py
from fastapi import APIRouter
from app.models.query import QueryRequest
from app.services.embedder_service import embedder
from app.services.ollama_service import call_ollama_specialize
from app.services.vllm_service import call_vllm
from app.core.settings import MODEL_NAME, TOPIC_TO_MODEL

router = APIRouter()

@router.post("/classify")
def classify(req: QueryRequest):
    results = embedder(req.text, top_k=req.top_k)
    return {"topics": results}

@router.post("/create_prompts")
def create_prompts(req: QueryRequest):
    topics = embedder(req.text, top_k=req.top_k)
    if not topics:
        return {"original": req.text, "topics": [], "prompts": {}}

    prompts = {}
    for t in topics:
        if float(t.get("score", 0.0)) >= 0.20:
            specialized = call_ollama_specialize(t["topic"], req.text)
            prompts[t["topic"]] = {"score": t["score"], "prompt": specialized}

    return {"original": req.text, "model": MODEL_NAME, "topics": topics, "prompts": prompts}

@router.post("/run_experts")
def run_expert_models(req: QueryRequest):
    topics = embedder(req.text, top_k=req.top_k)
    prompts, results = {}, {}

    for t in topics:
        topic_name = t["topic"]
        score = float(t.get("score", 0.0))
        if score < 0.20:
            continue

        specialized = call_ollama_specialize(topic_name, req.text)
        prompts[topic_name] = {"score": score, "prompt": specialized}

        model_id = TOPIC_TO_MODEL.get(topic_name)
        if model_id:
            results[topic_name] = call_vllm(model_id, specialized)
        else:
            results[topic_name] = f"[ERROR] No model for topic '{topic_name}'"

    return {"original": req.text, "model": MODEL_NAME, "topics": topics, "prompts": prompts, "results": results}
