from pydantic import BaseModel, Field

class ExpertConfig(BaseModel):
    name: str
    model_id: str
    max_new_tokens: str
    temperature: str
    node_port: int