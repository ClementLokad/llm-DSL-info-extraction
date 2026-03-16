"""
Agents package for LLMs
Provides modular, plugin-like architecture for easy extension.
"""

# Import specific agents (for backward compatibility)
from .mistral_agent import MistralAgent  
from .groq_agent import GroqAgent
from .qwen_agent import QwenAgent
from .qwen_ssh_agent import QwenSSHAgent

__all__ = [
    "MistralAgent", "GroqAgent", "QwenAgent", "QwenSSHAgent"
]