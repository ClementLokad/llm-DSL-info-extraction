import ollama
from .base import LLMAgent
from typing import Optional, Any

class QwenAgent(LLMAgent):
    """Implementation of an agent using Qwen AI"""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct-q4_k_m"):
        """
        Initialize the Qwen agent.
        
        Args:
            model: The Qwen model to use (default: qwen2.5:7b-instruct-q4_k_m)
        """
        super().__init__()
        self._model = model
    
    def initialize(self):
        """No itialization needed for Qwen via ollama"""
        return super().initialize()
     
    def generate_response(self, question: str, context: Optional[Any] = None) -> str:
        """
        Generate a response using Qwen.
        
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
            options={"num_ctx": 8192}
        )
        
        self.context = response["context"]
        
        return response['response']
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return f"Qwen-{self._model}"

if __name__ == "__main__":
    agent = QwenAgent()
    question = "What is the capital of France?"
    response = agent.generate_response(question)
    print(f"Q: {question}\nA: {response}")