"""
Hugging Face Transformers embedder implementation.
Example of how easy it is to add a new embedder to the system.
"""
from typing import List
from .base import BaseEmbedder

class HuggingFaceEmbedder(BaseEmbedder):
    """Hugging Face transformers embedder implementation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._model_name = model_name
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "huggingface"