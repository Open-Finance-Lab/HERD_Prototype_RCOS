import requests
from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.services.embedder_service import embedder
from app.services.ollama_service import call_ollama_specialize
from app.services.local_model_service import run_local_model
from app.core.settings import MODEL_NAME, TOPIC_TO_MODEL, OLLAMA_URL, TEMP, NUM_PREDICT, TIMEOUT_SECS


router = APIRouter()

@router.post("/classify")
def classify(req: QueryRequest):
    results = embedder(req.text, top_k=req.top_k)
    return {"topics": results}

@router.post("/create_prompts")
def create_prompts(req: QueryRequest):
    print(TOPIC_TO_MODEL["Physics"])
    topics = embedder(req.text, top_k=req.top_k)
    if not topics:
        return {"original": req.text, "topics": [], "prompts": {}}

    prompts = {}
    for t in topics:
        if float(t.get("score", 0.0)) >= 0.20:
            specialized = call_ollama_specialize(t["topic"], req.text)
            prompts[t["topic"]] = {"score": t["score"], "prompt": specialized}

    return {"original": req.text, "model": MODEL_NAME, "topics": topics, "prompts": prompts}

@router.post("/run_experts_locally")
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

        model_path = TOPIC_TO_MODEL.get(topic_name)
        if model_path:
            results[topic_name] = run_local_model(model_path, specialized)
        else:
            results[topic_name] = f"[ERROR] No model for topic '{topic_name}'"

    return {
        "original": req.text,
        "model": MODEL_NAME,
        "topics": topics,
        "prompts": prompts,
        "results": results
    }

@router.post("/Call_experts_locally_and_aggregate")
def call_experts_locally_and_aggregate(req: QueryRequest):
    """
    Full pipeline:
    1) Get topics
    2) Create specialized prompts (Ollama)
    3) Run per-topic local experts
    4) Aggregate final answer (Ollama) using all expert answers + original question
    """
    # 1) classify
    topics = embedder(req.text, top_k=req.top_k)

    # 2) specialized prompts
    prompts = {}
    results = {}
    for t in topics:
        topic_name = t["topic"]
        score = float(t.get("score", 0.0))
        if score < 0.20:
            continue

        specialized = call_ollama_specialize(topic_name, req.text)
        prompts[topic_name] = {"score": score, "prompt": specialized}

        # 3) run the local expert model for this topic
        model_path = TOPIC_TO_MODEL.get(topic_name)
        if model_path:
            try:
                results[topic_name] = run_local_model(model_path, specialized)
            except Exception as e:
                results[topic_name] = f"[ERROR] Local model for '{topic_name}' failed: {e}"
        else:
            results[topic_name] = f"[ERROR] No model for topic '{topic_name}'"

    expert_context_lines = []
    for topic_name, data in prompts.items():
        ans = results.get(topic_name, "[NO RESULT]")
        expert_context_lines.append(
            f"### Topic: {topic_name}\n"
            f"- Relevance score: {data['score']:.3f}\n"
            f"- Specialized prompt:\n{data['prompt']}\n"
            f"- Expert answer:\n{ans}\n"
        )
    expert_context = "\n\n".join(expert_context_lines) if expert_context_lines else "[No expert answers available]"

    system_aggregator = (
        "You are the Final Aggregator. Combine multiple expert answers into a single, "
        "coherent, non-redundant solution. Resolve conflicts with justification, cite "
        "which topic informed each key step (e.g., [Physics], [Math]), and surface any "
        "assumptions or uncertainties. Return:\n"
        "1) Final Answer (concise, actionable)\n"
        "2) Reasoning Summary (short; conflict resolution if any)\n"
        "3) Assumptions & Uncertainties\n"
        "4) Sources: list the contributing topics"
    )

    aggregation_user_message = (
        f"ORIGINAL QUESTION:\n{req.text}\n\n"
        f"EXPERT CONTEXT (specialized prompts + answers):\n{expert_context}\n\n"
        "Instructions:\n"
        "- Use the expert answers above as authoritative context.\n"
        "- If experts disagree, explain and choose the most rigorous answer.\n"
        "- Do not invent facts beyond context unless clearly marked as assumptions.\n"
        "- Keep the Final Answer up front and crisp."
    )

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "options": {
            "temperature": min(0.7, max(0.0, TEMP)),  
            "num_predict": NUM_PREDICT,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.05,
        },
        "messages": [
            {"role": "system", "content": system_aggregator},
            {"role": "user", "content": aggregation_user_message},
        ],
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        final_answer = r.json()["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama aggregation error: {e}")

    return {
        "original": req.text,
        "model": MODEL_NAME,
        "topics": topics,          
        "prompts": prompts,        
        "results": results,        
        "final_answer": final_answer
    }

    
