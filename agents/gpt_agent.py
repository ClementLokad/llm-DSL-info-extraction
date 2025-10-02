from typing import Optional
import openai
from .base import LLMAgent, rate_limited
import os


class GPTAgent(LLMAgent):
    """Implementation of an agent using OpenAI's GPT"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the GPT agent.
        
        Args:
            model: The GPT model to use (default: gpt-4o-mini)
        """
        super().__init__()
        self.client = None
        self.api_key = None
        self._model = model
        
    def initialize(self) -> None:
        """
        Initialize the connection to the OpenAI API.
        
        Raises:
            ValueError: If the API key is missing or invalid
            RuntimeError: If the connection to the API fails
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        try:
            self.client = openai.Client(api_key=self.api_key)
            # Test connection with a simple request
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    @rate_limited()
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using GPT.
        
        Args:
            question: The question to ask
            context: Optional context to provide
            
        Returns:
            str: The generated response
        """
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        # Prepare messages
        messages = []
        
        if context:
            messages.append({
                "role": "system", 
                "content": f"Use the following context to answer the question:\n{context}"
            })
            
        messages.append({"role": "user", "content": question})
        
        # Call GPT API
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return f"GPT-{self._model}"