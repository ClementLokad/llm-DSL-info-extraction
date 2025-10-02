"""
Agents package for LLMs
Provides modular, plugin-like architecture for easy extension.
"""

# Import specific agents (for backward compatibility)
from .gpt_agent import GPTAgent
from .mistral_agent import MistralAgent  
from .gemini_agent import GeminiAgent