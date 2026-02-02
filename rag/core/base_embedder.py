"""
Base embedder interface for generating code embeddings.

This module defines the abstract interface for embedding code chunks into
vector representations suitable for semantic search and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from rag.utils.handle_tokens import get_token_count
import numpy as np
import logging

from rag.core.base_chunker import CodeChunk

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """
    Abstract base class for all code embedders.
    
    This class defines the interface for converting code chunks into vector
    embeddings suitable for semantic search and similarity matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedder with configuration.
        
        Args:
            config: Embedder-specific configuration options
                   Common options:
                   - model_name: Name/path of embedding model
                   - batch_size: Number of chunks to process at once
                   - max_length: Maximum token length for input
                   - normalize: Whether to normalize embeddings
        """
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'default')
        self.batch_size = self.config.get('batch_size', 32)
        self.max_length = self.config.get('max_length', 512)
        self.normalize = self.config.get('normalize', True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Return the dimension of embeddings produced by this embedder.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the embedding model and any required resources.
        
        This method should load the model, set up API connections, etc.
        Should be called before using embed_chunks or embed_text.
        
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """
        Generate embeddings for a list of code chunks.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            NumPy array of shape (len(chunks), embedding_dimension)
            
        Raises:
            RuntimeError: If embedder is not initialized
            ValueError: If chunks are invalid or too long
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed (e.g., user query)
            
        Returns:
            NumPy array of shape (embedding_dimension,)
            
        Raises:
            RuntimeError: If embedder is not initialized
            ValueError: If text is invalid or too long
        """
        pass
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of text strings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of shape (len(texts), embedding_dimension)
            
        Raises:
            RuntimeError: If embedder is not initialized
            ValueError: If texts are invalid
        """
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        if not texts:
            return np.array([]).reshape(0, self.embedding_dimension)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_impl(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    @abstractmethod
    def _embed_batch_impl(self, texts: List[str]) -> np.ndarray:
        """
        Implementation-specific batch embedding method.
        
        Args:
            texts: Batch of texts to embed (size <= self.batch_size)
            
        Returns:
            NumPy array of embeddings
        """
        pass
    
    def prepare_text_for_embedding(self, text: str) -> str:
        """
        Prepare text for embedding with smart cleaning and truncation.
        
        Args:
            text: Raw text to prepare
            
        Returns:
            Prepared text ready for embedding
        """
        if not text or not text.strip():
            return ""
        
        # Get configuration parameters
        from config_manager import get_config
        config = get_config()
        truncation_ratio = config.get('embedder.text_preparation.truncation_ratio', 0.8)
        min_lines = config.get('embedder.text_preparation.min_lines_preserve', 1)
        
        # 1. Smart cleaning: preserve code structure but remove noise
        # Remove empty lines but keep indentation for code structure
        lines = [line.rstrip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 2. Better token estimation for code (more tokens per char than prose)
        estimated_tokens = get_token_count(text)
        
        # 3. Smart truncation at logical boundaries if needed
        if estimated_tokens > self.max_length:
            lines = text.split('\n')
            # Calculate target lines based on current token ratio
            target_lines = int(len(lines) * truncation_ratio * self.max_length / estimated_tokens)
            # Ensure we keep at least min_lines, at most all lines
            target_lines = max(min_lines, min(target_lines, len(lines)))
            text = '\n'.join(lines[:target_lines])
            
            self.logger.debug(f"Truncated text from {estimated_tokens} to ~{self.max_length} tokens")
        
        return text
    
    def prepare_chunk_for_embedding(self, chunk: CodeChunk) -> str:
        """
        Prepare a code chunk for embedding.
        
        Args:
            chunk: Code chunk to prepare
            
        Returns:
            Prepared text representation of the chunk
        """
        # Use summary from metadata if available
        if "summary" in chunk.metadata and chunk.metadata["summary"]:
            return self.prepare_text_for_embedding(chunk.metadata["summary"])

        # Default implementation - can be overridden by specific embedders
        text_parts = []
        
        # Add the main content
        text_parts.append(chunk.content)
        
        # Join with newlines and prepare
        full_text = '\n'.join(text_parts)
        return self.prepare_text_for_embedding(full_text)
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding has the correct shape and properties.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if not isinstance(embedding, np.ndarray):
            self.logger.error("Embedding must be a NumPy array")
            return False
        
        if len(embedding.shape) != 1:
            self.logger.error(f"Embedding must be 1-dimensional, got shape {embedding.shape}")
            return False
        
        if embedding.shape[0] != self.embedding_dimension:
            self.logger.error(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {embedding.shape[0]}")
            return False
        
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            self.logger.error("Embedding contains NaN or infinite values")
            return False
        
        return True
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (cosine similarity by default)
        """
        # Cosine similarity
        if self.normalize:
            # If embeddings are normalized, dot product = cosine similarity
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity manually
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
