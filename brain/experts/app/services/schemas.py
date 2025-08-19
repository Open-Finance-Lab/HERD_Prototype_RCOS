from pydantic import BaseModel, Field

class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    stop: list[str] | None = None

class InferResponse(BaseModel):
    model_id: str
    completion: str
