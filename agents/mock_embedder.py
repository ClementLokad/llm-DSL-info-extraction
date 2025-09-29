"""
Mock embedder for testing without API calls.
"""
import random
from typing import List
from .base import BaseEmbedder

class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing without API calls."""
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        random.seed(42)  # For reproducible results
    
    def embed(self, text: str) -> List[float]:
        # Generate consistent embeddings based on text hash
        text_hash = hash(text)
        random.seed(text_hash)
        return [random.uniform(-1, 1) for _ in range(self._dimension)]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "mock"