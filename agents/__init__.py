"""
Agents package for LLMs
Provides modular, plugin-like architecture for easy extension.
"""

# Import specific agents (for backward compatibility)
from .gpt_agent import GPTAgent
from .mistral_agent import MistralAgent  
from .gemini_agent import GeminiAgent
from .groq_agent import GroqAgent
from .llama3_agent import Llama3Agent
from .qwen_agent import QwenAgent

__all__ = [
    "GPTAgent", "MistralAgent", "GeminiAgent", "GroqAgent", "Llama3Agent", "QwenAgent"
]