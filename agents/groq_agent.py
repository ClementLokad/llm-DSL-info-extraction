from typing import Optional
import os
from groq import Groq
from .base import LLMAgent, rate_limited
from config_manager import get_config

class GroqAgent(LLMAgent):
    """Implementation of an agent using Groq"""
    
    def __init__(self, model: str = "qwen/qwen3-32b"):
        """
        Initialize the Groq agent.
        
        Args:
            model: The Groq model to use (default: qwen/qwen3-32b)
        """
        super().__init__()
        self.client = None
        self.api_key = None
        self._model = model
        
    @property
    def model_name(self) -> str:
        """Return the name of the model used by the agent."""
        return self._model

    def initialize(self) -> None:
        """
        Initialize the connection to the Groq API.
        
        Raises:
            ValueError: If the API key is missing or invalid
            RuntimeError: If the connection to the API fails
        """
        config = get_config()
        self.api_key = config.get_api_key('GROQ_API_KEY')
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        try:
            self.client = Groq(api_key=self.api_key)
            # Simple test call to verify connection/key
            # We'll just list models to verify auth
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Groq API: {str(e)}")
    
    @rate_limited()
    @LLMAgent.count_tokens
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using Groq.
        
        Args:
            question: The question to answer
            context: Optional context to include
            
        Returns:
            The generated response
        """
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        messages = []
        
        if context:
            system_prompt = (
                "You are an expert assistant for Envision DSL code analysis. "
                "Use the provided context to answer the user's question. "
                "If the answer cannot be found in the context, say so."
            )
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            system_prompt = "You are an expert assistant for Envision DSL code analysis."
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self._model,
                temperature=0.1,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
