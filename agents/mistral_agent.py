from typing import Optional, List, Dict
import requests
from .base import LLMAgent, rate_limited
from config_manager import get_config


class MistralAgent(LLMAgent):
    """Implementation of an agent using Mistral AI"""
    
    def __init__(self, model: str = "mistral-large-latest"):
        """
        Initialize the Mistral agent.
        
        Args:
            model: The Mistral model to use (default: mistral-large-latest)
        """
        super().__init__()
        self.base_url = "https://api.mistral.ai/v1"
        self.api_key = None
        self._model = model
        
    def initialize(self) -> None:
        """
        Initialize the connection to the Mistral API.
        
        Raises:
            ValueError: If the API key is missing or invalid
            RuntimeError: If the connection to the API fails
        """
        config = get_config()
        self.api_key = config.get_api_key('MISTRAL_API_KEY')
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
        # Test the API key with a simple request
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Mistral API: {str(e)}")
    
    @rate_limited()        
    def generate_response(self, question: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using Mistral.
        
        Args:
            question: The question to ask
            context: Optional context to provide
            
        Returns:
            str: The generated response
        """
        if not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # 1. Handle Initial System/RAG Context
        # If 'context' is provided and history is empty, set it as the System Prompt.
        # This static prefix is crucial for caching efficiency.
        if context:
            self.context = context
        else:
            self.context = []

        # 2. Append the new User Question to history
        self.context.append({"role": "user", "content": question})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 3. Send the FULL history (self.context) to the API
        payload = {
            "model": self._model,
            "messages": self.context,  # <--- This is where the cache magic happens
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        assistant_content = result["choices"][0]["message"]["content"]

        # 4. Append the Assistant Response to history
        # This ensures the NEXT request starts with [System + User + Assistant],
        # allowing Mistral to reuse the KV cache for this entire block.
        self.context.append({"role": "assistant", "content": assistant_content})

        return assistant_content
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return f"Mistral-{self._model}"