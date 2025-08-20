import json
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore

class Embedder:
    def __init__(self, vector_file: str, model_name: str = "intfloat/e5-small-v2"):
        self.model = SentenceTransformer(model_name)
        with open(vector_file, "r") as f:
            self.topic_vectors = json.load(f)

        self.topic_ids = list(self.topic_vectors.keys())
        vecs = [np.array(self.topic_vectors[t], dtype=np.float32) for t in self.topic_ids]
        vecs = np.vstack(vecs)
        self.topic_matrix = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed and L2-normalize a single text string."""
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec[0]  

    def __call__(self, query: str, top_k: int = 3):
        """Embed query, compare to stored topics, return top-k matches."""
        q_vec = self.embed_text(query)
        sims = self.topic_matrix @ q_vec  

        order = np.argsort(-sims)  
        top_topics = [
            {"topic": self.topic_ids[i], "score": float(sims[i])}
            for i in order[:top_k]
        ]
        return top_topics