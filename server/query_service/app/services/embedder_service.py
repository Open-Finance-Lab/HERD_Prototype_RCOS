import os
from pathlib import Path
from modules.embedder.embedder import Embedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOPIC_VECTOR = os.getenv("TOPIC_VECTOR", "topic_vector.json")

vector_path = DATA_DIR / TOPIC_VECTOR

embedder = Embedder(vector_path, model_name="all-MiniLM-L6-v2")