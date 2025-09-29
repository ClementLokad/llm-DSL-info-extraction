"""
Module for handling text tokenization using appropriate tokenizer based on the model.
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai

class BaseTokenizer(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text."""
        pass

class GeminiTokenizer(BaseTokenizer):
    """Tokenizer for Gemini models."""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using Gemini's tokenizer."""
        # Note: Currently Gemini API doesn't provide direct access to tokenization
        # This is a simplified implementation
        return text.split()
        
    def get_token_count(self, text: str) -> int:
        """Estimate token count for Gemini models."""
        # This is a rough estimation as Gemini doesn't expose token counting
        # Using average word-to-token ratio of 1.3
        words = len(text.split())
        return int(words * 1.3)

class TokenizerFactory:
    """Factory for creating appropriate tokenizer instances."""
    
    @staticmethod
    def get_tokenizer(model_type: str, **kwargs) -> BaseTokenizer:
        """Get the appropriate tokenizer for the given model type."""
        tokenizers = {
            'gemini': GeminiTokenizer,
            # Add more tokenizers as needed
        }
        
        tokenizer_class = tokenizers.get(model_type.lower())
        if not tokenizer_class:
            raise ValueError(f"No tokenizer available for model type: {model_type}")
            
        return tokenizer_class(**kwargs)