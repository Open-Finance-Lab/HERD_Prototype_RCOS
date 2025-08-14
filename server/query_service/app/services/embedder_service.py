from pathlib import Path
from embedding.embedder import Embedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
vector_path = DATA_DIR / "topic_vector.json"

embedder = Embedder(vector_path, model_name="all-MiniLM-L6-v2")