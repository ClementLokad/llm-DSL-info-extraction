from typing import List, Dict, Any, Optional
import openai
from .base import LLMAgent
import os

class GPTAgent(LLMAgent):
    """Implementation of an agent using OpenAI's GPT"""
    
    def __init__(self):
        self.client = None
        
    def initialize(self) -> None:
        """Initialize the OpenAI client with the API key"""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.Client()
        
    def process_question(self, question: str, context: Optional[List[str]] = None) -> str:
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        # Prepare context
        messages = []
        if context:
            context_text = "\n".join(context)
            messages.append({"role": "system", "content": f"Code context:\n{context_text}"})
            
        messages.append({"role": "user", "content": question})
        
        # Call GPT API
        response = self.client.chat.completions.create(
            model="gpt-4",  # or other model as needed
            messages=messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
        
    def get_embedding(self, text: str) -> List[float]:
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        return response.data[0].embedding
        
    def extract_metadata(self, code_chunk: str) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        prompt = f"""Analyze this code chunk and extract metadata:
        {code_chunk}
        
        Return the metadata as a JSON with:
        - functions: list of function names
        - variables: list of important variables
        - dependencies: list of imports/dependencies
        - description: brief description of what the code does"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # We want a deterministic response
        )
        
        # Note: In a real case, we should properly parse the JSON response
        return eval(response.choices[0].message.content)