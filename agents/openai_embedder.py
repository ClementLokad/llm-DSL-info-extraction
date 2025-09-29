"""
OpenAI embedder implementation.
"""
from typing import List
from .base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dimension = 1536
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "openai"