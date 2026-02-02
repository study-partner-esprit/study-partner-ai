
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    Wrapper for creating vector embeddings from text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of texts into embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)