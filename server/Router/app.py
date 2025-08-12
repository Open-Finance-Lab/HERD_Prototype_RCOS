from fastapi import FastAPI
from pydantic import BaseModel
from embedding.embedder import Embedder

embedder = Embedder("topic_vector.json", model_name="all-MiniLM-L6-v2")

app = FastAPI()

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3

@app.post("/classify")
def classify(req: QueryRequest):
    results = embedder(req.text, top_k=req.top_k)
    return {"topics": results}