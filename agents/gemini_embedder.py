"""
Google Gemini embedder implementation.
"""
from typing import List
from .base import BaseEmbedder

class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedder implementation."""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self._model_name = "models/embedding-001"
        self._dimension = 768
    
    def embed(self, text: str) -> List[float]:
        response = self.genai.embed_content(model=self._model_name, content=text)
        return response['embedding']
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing embedding {i+1}/{len(texts)}")
            embeddings.append(self.embed(text))
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "gemini"