from fastapi import APIRouter, Depends
from app.services.schemas import InferRequest, InferResponse
from app.services.inference import InferenceService
from app.core.config import settings

router = APIRouter()
inference_service = InferenceService()

@router.get("/healthz")
def healthz():
    return {"status": "ok", "model_id": inference_service.model_id, "is_local": settings.IS_LOCAL}

@router.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    params = {
        "max_new_tokens": req.max_new_tokens or settings.MAX_NEW_TOKENS,
        "temperature": req.temperature if req.temperature is not None else settings.TEMPERATURE,
        "top_p": req.top_p if req.top_p is not None else settings.TOP_P,
        "do_sample": True,
    }
    if (req.top_k or settings.TOP_K) > 0:
        params["top_k"] = req.top_k or settings.TOP_K
    if req.repetition_penalty or settings.REPETITION_PENALTY != 1.0:
        params["repetition_penalty"] = req.repetition_penalty or settings.REPETITION_PENALTY
    if req.stop or settings.STOP:
        params["eos_token_id"] = None
        params["stop_sequence"] = req.stop or settings.STOP

    out = inference_service.infer(req.prompt, **params)
    return InferResponse(model_id=inference_service.model_id, completion=out)
