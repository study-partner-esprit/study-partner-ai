#building vector store for course materials
import faiss
import numpy as np

class VectorStore:
    """
    Stores embeddings and allows similarity search
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: list[str]):
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        self.index.add(normalized_embeddings)
        self.texts.extend(texts)

    def search(self, query_emb: np.ndarray, k: int = 5):
        # Normalize query embedding
        norm = np.linalg.norm(query_emb)
        if norm == 0:
            normalized_query = query_emb
        else:
            normalized_query = query_emb / norm
        distances, indices = self.index.search(normalized_query.reshape(1, -1), k)
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        return results