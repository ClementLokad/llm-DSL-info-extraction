"""
Configuration management for model settings and paths.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    embedding_dim: int
    context_length: int
    model_type: str  # e.g., 'gemini', 'gpt', 'mistral'
    additional_params: Dict[str, Any] = None

    def get_index_path(self, base_dir: str) -> str:
        """Get the path for storing this model's index."""
        return str(Path(base_dir) / f"{self.model_type}_{self.name}_index")

    @property
    def identifier(self) -> str:
        """Get unique identifier for this model configuration."""
        return f"{self.model_type}_{self.name}"

class ModelRegistry:
    """Registry for available models and their configurations."""
    
    _default_configs = {
        'gemini': {
            'models/gemini-2.5-pro': ModelConfig(
                name='models/gemini-2.5-pro',
                embedding_dim=768,
                context_length=32768,
                model_type='gemini'
            )
        },
        'gpt': {
            'gpt-4': ModelConfig(
                name='gpt-4',
                embedding_dim=1536,
                context_length=8192,
                model_type='gpt'
            ),
            'gpt-3.5-turbo': ModelConfig(
                name='gpt-3.5-turbo',
                embedding_dim=1536,
                context_length=4096,
                model_type='gpt'
            )
        },
        'mistral': {
            'mistral-7b': ModelConfig(
                name='mistral-7b',
                embedding_dim=4096,
                context_length=8192,
                model_type='mistral'
            )
        }
    }

    def __init__(self):
        self._configs = self._default_configs.copy()
        self._active_config: Optional[ModelConfig] = None

    def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        if config.model_type not in self._configs:
            self._configs[config.model_type] = {}
        self._configs[config.model_type][config.name] = config

    def get_config(self, model_type: str, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        try:
            return self._configs[model_type][model_name]
        except KeyError:
            raise ValueError(f"No configuration found for {model_type}/{model_name}")

    def set_active_config(self, model_type: str, model_name: str) -> None:
        """Set the active model configuration."""
        self._active_config = self.get_config(model_type, model_name)

    @property
    def active_config(self) -> ModelConfig:
        """Get the currently active model configuration."""
        if not self._active_config:
            raise ValueError("No active model configuration set")
        return self._active_config

    def list_available_models(self) -> Dict[str, list]:
        """List all available model configurations."""
        return {
            model_type: list(configs.keys())
            for model_type, configs in self._configs.items()
        }

# Global registry instance
registry = ModelRegistry()