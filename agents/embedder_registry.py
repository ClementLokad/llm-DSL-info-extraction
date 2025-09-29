"""
Dynamic embedder loading system.
Automatically discovers and loads embedder classes from the agents directory.
"""
import os
import importlib
from typing import Dict, Type, Optional
from .base import BaseEmbedder

class EmbedderRegistry:
    """Registry for dynamically loading embedder classes."""
    
    def __init__(self):
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._discover_embedders()
    
    def _discover_embedders(self):
        """Automatically discover embedder classes in the agents directory."""
        agents_dir = os.path.dirname(__file__)
        
        for filename in os.listdir(agents_dir):
            if filename.endswith("_embedder.py") and not filename.startswith("__"):
                module_name = filename[:-3]  # Remove .py
                try:
                    module = importlib.import_module(f".{module_name}", package="agents")
                    
                    # Look for classes that inherit from BaseEmbedder
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseEmbedder) and 
                            attr != BaseEmbedder):
                            
                            # Register embedder by its model_name
                            embedder_instance = attr.__new__(attr)  # Create dummy instance to get model_name
                            if hasattr(embedder_instance, '_get_model_name'):
                                model_name = embedder_instance._get_model_name()
                            else:
                                # Extract model name from class name (e.g., GeminiEmbedder -> gemini)
                                model_name = attr.__name__.replace("Embedder", "").lower()
                            
                            self._embedders[model_name] = attr
                            print(f"Discovered embedder: {model_name} -> {attr.__name__}")
                            
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
    
    def get_embedder(self, model_name: str, **kwargs) -> BaseEmbedder:
        """Get an embedder instance by model name."""
        if model_name not in self._embedders:
            available = list(self._embedders.keys())
            raise ValueError(f"Unknown embedder '{model_name}'. Available: {available}")
        
        embedder_class = self._embedders[model_name]
        return embedder_class(**kwargs)
    
    def list_available(self) -> list:
        """List all available embedder names."""
        return list(self._embedders.keys())

# Global registry instance
_registry = EmbedderRegistry()

def get_embedder(model_name: str, **kwargs) -> BaseEmbedder:
    """Convenience function to get an embedder."""
    return _registry.get_embedder(model_name, **kwargs)

def list_available_embedders() -> list:
    """List all available embedder names."""
    return _registry.list_available()