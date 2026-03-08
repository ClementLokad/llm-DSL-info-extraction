"""
Configuration management system for the LLM DSL Info Extraction project.

This module provides a centralized way to load and access configuration from:
- .env files (secrets and environment variables)
- YAML files (application configuration and parameters)

All numeric constants and parameters are externalized to configuration files
to enable easy adjustment without code changes.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager that loads settings from multiple sources.
    
    Precedence (highest to lowest):
    1. Environment variables (from .env file)
    2. YAML configuration file
    3. Default values defined in YAML
    
    Usage:
        config = ConfigManager()
        
        # Access configuration sections
        chunker_config = config.get_chunker_config()
        embedder_config = config.get_embedder_config()
        
        # Access specific values
        max_tokens = config.get('chunker.max_chunk_tokens')
        api_key = config.get_api_key('gemini')
    """
    
    def __init__(self, config_file: str = "config.yaml", env_file: str = ".env"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
            env_file: Path to environment variables file
        """
        self.config_file = Path(config_file)
        self.env_file = Path(env_file)
        self.config: Dict[str, Any] = {}
        
        # Load configuration
        self._load_environment()
        self._load_yaml_config()
        self._validate_config()
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.warning(f"Environment file {self.env_file} not found")
    
    def _load_yaml_config(self):
        """Load application configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _validate_config(self):
        """Validate essential configuration sections exist."""
        required_sections = ['parser', 'chunker', 'embedder', 'retriever']
        missing_sections = [section for section in required_sections 
                          if section not in self.config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'chunker.max_chunk_tokens')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Examples:
            >>> config.get('chunker.max_chunk_tokens')
            512
            >>> config.get('embedder.sentence_transformer.model_name')
            'all-MiniLM-L6-v2'
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider from environment variables.
        
        Args:
            provider: Provider name (e.g. 'GOOGLE_API_KEY', 'OPENAI_API_KEY')
            
        Returns:
            API key or None if not found
        """
        
        # First try as provider name
        env_key = os.getenv(provider)
        if env_key:
            return env_key
        
        logger.warning(f"Unknown provider: {provider}")
        return None
    
    def get_parser_config(self) -> Dict[str, Any]:
        """Get parser configuration section."""
        return self.config.get('parser', {})
    
    def get_chunker_config(self) -> Dict[str, Any]:
        """Get chunker configuration section."""
        return self.config.get('chunker', {})
    
    def get_summarizer_config(self) -> Dict[str, Any]:
        """Get summarizer configuration section."""
        return self.config.get('summarizer', {})
    
    def get_query_transformer_config(self) -> Dict[str, Any]:
        """Get query transformer configuration section."""
        return self.config.get('query_transformer', {})
    
    def get_embedder_config(self, embedder_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get embedder configuration section.
        
        Args:
            embedder_type: Specific embedder type or None for default
            
        Returns:
            Embedder configuration
        """
        embedder_config = self.config.get('embedder', {})
        
        if embedder_type is None:
            embedder_type = embedder_config.get('default_type', 'sentence_transformer')
        
        # Return both general config and specific embedder config
        specific_config = embedder_config.get(embedder_type, {})
        return {
            'type': embedder_type,
            'general': embedder_config,
            'specific': specific_config,
            **specific_config  # Flatten specific config for easier access
        }
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get retriever configuration section."""
        retriever_config = self.config.get('retriever', {})
        
        # Flatten FAISS-specific config to top level for compatibility
        faiss_config = retriever_config.get('faiss', {})
        if faiss_config:
            retriever_config = {**retriever_config, **faiss_config}
        
        return retriever_config
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration section."""
        return self.config.get('pipeline', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config.get('logging', {})
    
    def get_default_agent(self) -> str:
        """Get default agent from configuration."""
        return self.config.get('agent', {}).get('default_model', 'mistral')
    
    def get_summary_agent(self) -> str:
        """Get agent for chunk summaries. Use summary_model if specified, otherwise default."""
        return self.config.get('summarizer', {}).get('summary_model', self.get_default_agent())

    def get_benchmark_agent(self) -> str:
        """Get agent for llm as a judge. Use benchmark_model if specified, otherwise default."""
        return self.config.get('benchmark', {}).get('benchmark_model', self.get_default_agent())
    
    def get_query_transformer_agent(self) -> str:
        """Get agent for query transformers (hyde, fusion etc). Use query_transformer_model if specified, otherwise default."""
        return self.config.get('query_transformer', {}).get('query_transformer_model', self.get_default_agent())
    
    def get_benchmark_type(self) -> str:
        """Get benchmark type from configuration."""
        return self.config.get('benchmark', {}).get('benchmark_type', 'cosine_similarity')
        
    def __repr__(self) -> str:
        """String representation of configuration manager."""
        return f"ConfigManager(env=dev, config_file={self.config_file})"

# Global configuration instance
_config_instance: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """
    Get global configuration instance (singleton pattern).
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def reload_config() -> ConfigManager:
    """
    Reload configuration (useful for testing or configuration changes).
    
    Returns:
        New ConfigManager instance
    """
    global _config_instance
    _config_instance = ConfigManager()
    return _config_instance