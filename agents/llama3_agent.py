import ollama
from .base import LLMAgent
from typing import Optional

class Llama3Agent(LLMAgent):
    """Implementation of an agent using Llama3 AI"""
    
    def __init__(self, model: str = "llama3:latest"):
        """
        Initialize the Llama3 agent.
        
        Args:
            model: The Llama3 model to use (default: llama3:latest)
        """
        super().__init__()
        self.model = model
    
    def initialize(self):
        """No itialization needed for Llama3 via ollama"""
        return super().initialize()
    
    @LLMAgent.count_tokens
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using Llama3.
        
        Args:
            question: The question to ask
            context: Optional context to provide
            
        Returns:
            str: The generated response
        """
        
        prompt = question

        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            context=context,
            stream=False, # Request the full response at once
            options={"num_ctx": 16384}
        )
        
        self.context = response["context"]

        return response['response']
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return f"Llama3-{self._model}"

if __name__ == "__main__":
    agent = Llama3Agent()
    question = "What is the capital of France?"
    response = agent.generate_response(question)
    print(f"Q: {question}\nA: {response}")