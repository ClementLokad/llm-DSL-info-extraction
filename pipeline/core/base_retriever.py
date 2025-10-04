"""
Base retriever interface for semantic code search and retrieval.

This module defines the abstract interface for storing embeddings and
retrieving relevant code chunks based on similarity search.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from preprocessing.core.base_chunker import CodeChunk

logger = logging.getLogger(__name__)

class RetrievalResult:
    """
    Represents a single retrieval result with score and metadata.
    
    Attributes:
        chunk: The retrieved code chunk
        score: Similarity/relevance score
        rank: Rank in the result set (1-based)
        metadata: Additional retrieval-specific metadata
    """
    
    def __init__(self, chunk: CodeChunk, score: float, rank: int = 0, metadata: Optional[Dict[str, Any]] = None):
        self.chunk = chunk
        self.score = score
        self.rank = rank
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"RetrievalResult(score={self.score:.3f}, rank={self.rank}, chunk_type={self.chunk.chunk_type})"

class BaseRetriever(ABC):
    """
    Abstract base class for all code retrievers.
    
    This class defines the interface for storing code embeddings and
    retrieving relevant chunks based on similarity search.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retriever with configuration.
        
        Args:
            config: Retriever-specific configuration options
                   Common options:
                   - index_path: Path to save/load index files
                   - similarity_metric: Metric for similarity computation
                   - top_k: Default number of results to return
        """
        self.config = config or {}
        self.index_path = self.config.get('index_path', './index')
        self.similarity_metric = self.config.get('similarity_metric', 'cosine')
        self.default_top_k = self.config.get('top_k', 10)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        self._chunks = []  # Store chunks for retrieval
    
    @abstractmethod
    def initialize(self, embedding_dimension: int) -> None:
        """
        Initialize the retriever with the specified embedding dimension.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """
        Add code chunks and their embeddings to the retriever.
        
        Args:
            chunks: List of code chunks to add
            embeddings: NumPy array of embeddings, shape (len(chunks), embedding_dim)
            
        Raises:
            RuntimeError: If retriever is not initialized
            ValueError: If chunks and embeddings don't match
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Search for similar code chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (uses default_top_k if None)
            
        Returns:
            List of retrieval results sorted by relevance (best first)
            
        Raises:
            RuntimeError: If retriever is not initialized
            ValueError: If query embedding is invalid
        """
        pass
    
    def search_by_text(self, query_text: str, embedder, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Search for similar code chunks using query text.
        
        Args:
            query_text: Natural language query
            embedder: Embedder instance to convert text to embedding
            top_k: Number of results to return
            
        Returns:
            List of retrieval results sorted by relevance
            
        Raises:
            RuntimeError: If retriever is not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")
        
        # Convert text to embedding
        query_embedding = embedder.embed_text(query_text)
        
        # Perform search
        return self.search(query_embedding, top_k)
    
    def add_single_chunk(self, chunk: CodeChunk, embedding: np.ndarray) -> None:
        """
        Add a single code chunk and its embedding.
        
        Args:
            chunk: Code chunk to add
            embedding: Embedding vector for the chunk
        """
        self.add_chunks([chunk], embedding.reshape(1, -1))
    
    def get_chunk_count(self) -> int:
        """
        Get the number of chunks stored in the retriever.
        
        Returns:
            Number of stored chunks
        """
        return len(self._chunks)
    
    def get_chunks_by_type(self, chunk_type: str) -> List[CodeChunk]:
        """
        Get all chunks of a specific type.
        
        Args:
            chunk_type: Type of chunks to retrieve
            
        Returns:
            List of chunks matching the type
        """
        return [chunk for chunk in self._chunks if chunk.chunk_type == chunk_type]
    
    def filter_results_by_score(self, results: List[RetrievalResult], min_score: float) -> List[RetrievalResult]:
        """
        Filter retrieval results by minimum score threshold.
        
        Args:
            results: List of retrieval results
            min_score: Minimum score threshold
            
        Returns:
            Filtered list of results
        """
        return [result for result in results if result.score >= min_score]
    
    def filter_results_by_type(self, results: List[RetrievalResult], chunk_types: List[str]) -> List[RetrievalResult]:
        """
        Filter retrieval results by chunk types.
        
        Args:
            results: List of retrieval results
            chunk_types: List of allowed chunk types
            
        Returns:
            Filtered list of results
        """
        return [result for result in results if result.chunk.chunk_type in chunk_types]
    
    def rerank_results(self, results: List[RetrievalResult], rerank_fn) -> List[RetrievalResult]:
        """
        Re-rank retrieval results using a custom function.
        
        Args:
            results: List of retrieval results to re-rank
            rerank_fn: Function that takes a RetrievalResult and returns a new score
            
        Returns:
            Re-ranked list of results
        """
        # Apply re-ranking function
        for result in results:
            new_score = rerank_fn(result)
            result.metadata['original_score'] = result.score
            result.score = new_score
        
        # Sort by new scores and update ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    @abstractmethod
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save the current index to disk.
        
        Args:
            path: Path to save index (uses self.index_path if None)
            
        Raises:
            RuntimeError: If saving fails
        """
        pass
    
    @abstractmethod
    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load an index from disk.
        
        Args:
            path: Path to load index from (uses self.index_path if None)
            
        Raises:
            RuntimeError: If loading fails
            FileNotFoundError: If index file doesn't exist
        """
        pass
    
    def clear_index(self) -> None:
        """
        Clear all stored chunks and embeddings.
        """
        self._chunks = []
        # Subclasses should override to clear their specific index structures
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the stored index.
        
        Returns:
            Dictionary with statistics
        """
        chunk_types = {}
        total_tokens = 0
        
        for chunk in self._chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_tokens += chunk.size_tokens
        
        return {
            'total_chunks': len(self._chunks),
            'chunk_types': chunk_types,
            'total_tokens': total_tokens,
            'average_tokens_per_chunk': total_tokens / len(self._chunks) if self._chunks else 0
        }