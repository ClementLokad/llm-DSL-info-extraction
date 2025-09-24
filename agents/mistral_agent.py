from typing import List, Dict, Any, Optional
import requests
from .base import LLMAgent
import os

class MistralAgent(LLMAgent):
    """Implementation of an agent using Mistral AI"""
    
    def __init__(self):
        self.api_key = None
        self.base_url = "https://api.mistral.ai/v1"  # URL to be adjusted according to official documentation
        
    def initialize(self) -> None:
        """Initialize the agent with Mistral API key"""
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Utility method to make requests to Mistral API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{self.base_url}/{endpoint}", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
        
    def process_question(self, question: str, context: Optional[List[str]] = None) -> str:
        if not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        messages = []
        if context:
            context_text = "\n".join(context)
            messages.append({"role": "system", "content": f"Code context:\n{context_text}"})
            
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": "mistral-large-latest",  # to be adjusted based on available models
            "messages": messages
        }
        
        response = self._make_request("chat/completions", payload)
        return response["choices"][0]["message"]["content"]
        
    def get_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        payload = {
            "model": "mistral-embed",  # to be adjusted based on available models
            "input": text
        }
        
        response = self._make_request("embeddings", payload)
        return response["data"][0]["embedding"]
        
    def extract_metadata(self, code_chunk: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        prompt = f"""Analyze this code chunk and extract metadata:
        {code_chunk}
        
        Return the metadata as a JSON with:
        - functions: list of function names
        - variables: list of important variables
        - dependencies: list of imports/dependencies
        - description: brief description of what the code does"""
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        response = self._make_request("chat/completions", payload)
        # Note: In a real case, we should properly parse the JSON response
        return eval(response["choices"][0]["message"]["content"])