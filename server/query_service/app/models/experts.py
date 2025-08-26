from pydantic import BaseModel, Field
from typing import Optional

class ExpertConfig(BaseModel):
    name: str
    model_id: str
    max_new_tokens: str
    temperature: str
    node_port: int

class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0