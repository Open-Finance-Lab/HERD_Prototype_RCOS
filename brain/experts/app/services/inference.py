import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.core.config import settings

class InferenceService:
    def __init__(self):
        self.model_id, self.pipeline = self._build_pipeline()

    def _torch_dtype(self):
        if settings.TORCH_DTYPE_AUTO:
            return "auto"
        if settings.TORCH_DTYPE:
            from torch import bfloat16, float16, float32
            mapping = {
                "bfloat16": bfloat16, "bf16": bfloat16,
                "float16": float16, "fp16": float16,
                "float32": float32, "fp32": float32,
            }
            return mapping.get(settings.TORCH_DTYPE.lower(), "auto")
        return "auto"

    def _build_pipeline(self):
        mid = os.getenv("MODEL_ID", settings.MODEL_ID)
        local = settings.IS_LOCAL
        device = 0 if torch.cuda.is_available() else -1
        dm = "auto" if (settings.DEVICE_MAP_AUTO and torch.cuda.is_available()) else None

        tok = AutoTokenizer.from_pretrained(
            mid,
            revision=settings.REVISION,
            use_auth_token=settings.HF_TOKEN,
            trust_remote_code=settings.TRUST_REMOTE_CODE,
            local_files_only=local,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            mid,
            revision=settings.REVISION,
            use_auth_token=settings.HF_TOKEN,
            trust_remote_code=settings.TRUST_REMOTE_CODE,
            torch_dtype=self._torch_dtype(),
            device_map=dm,
            local_files_only=local,
        )
        return mid, pipeline("text-generation", model=mdl, tokenizer=tok, device=device)

    def infer(self, prompt: str, **gen_kwargs) -> str:
        out = self.pipeline(prompt, **gen_kwargs)[0]["generated_text"]
        return out
