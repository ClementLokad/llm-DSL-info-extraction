"""
Agents package for LLM and embedding models.
Provides modular, plugin-like architecture for easy extension.
"""

# Import base classes
from .base import BaseEmbedder, LLMAgent

# Import embedder registry system
from .embedder_registry import get_embedder, list_available_embedders

# Import specific agents (for backward compatibility)
from .gpt_agent import GPTAgent
from .mistral_agent import MistralAgent  
from .gemini_agent import GeminiAgent

__all__ = [
    # Base classes
    'BaseEmbedder', 'LLMAgent',
    
    # Embedder system
    'get_embedder', 'list_available_embedders',
    
    # LLM Agents
    'GPTAgent', 'MistralAgent', 'GeminiAgent'
]