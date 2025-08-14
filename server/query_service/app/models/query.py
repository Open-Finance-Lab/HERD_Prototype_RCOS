from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3