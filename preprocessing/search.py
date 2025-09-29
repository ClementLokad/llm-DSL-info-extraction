"""
Search and retrieval functionality for processed DSL chunks.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import pickle

from .config import ModelConfig, registry
from .embeddings import EmbeddingManager, EmbeddingData

@dataclass
class SearchResult:
    """Container for search results."""
    text: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: int

class DSLSearchEngine:
    """Search engine for DSL code chunks."""
    
    def __init__(self, storage_dir: str):
        """Initialize the search engine.
        
        Args:
            storage_dir: Directory containing processed data
        """
        self.storage_dir = Path(storage_dir)
        self.embedding_managers: Dict[str, EmbeddingManager] = {}
        self._load_available_indices()

    def _load_available_indices(self):
        """Load all available indices from storage directory."""
        if not self.storage_dir.exists():
            return

        # Look for index files
        for index_file in self.storage_dir.glob("*_index.index"):
            # Extract model identifier from filename
            identifier = index_file.stem.replace("_index", "")
            
            try:
                # Load the embedding manager
                index_path = str(index_file.with_suffix(''))
                self.embedding_managers[identifier] = EmbeddingManager.load(index_path)
            except Exception as e:
                print(f"Error loading index {index_file}: {e}")

    def search(self, 
              query: str,
              model_config: Optional[ModelConfig] = None,
              k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for relevant DSL chunks.
        
        Args:
            query: Search query
            model_config: Specific model configuration to use, defaults to active config
            k: Number of results to return
            filters: Optional filters for metadata fields
            
        Returns:
            List of SearchResult objects
        """
        config = model_config or registry.active_config
        manager = self.embedding_managers.get(config.identifier)
        
        if not manager:
            raise ValueError(f"No index found for model {config.identifier}")
            
        # Generate query embedding (placeholder - should use actual model)
        query_embedding = np.random.randn(config.embedding_dim)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in the index
        results = manager.search(query_embedding, k=k * 2)  # Get extra results for filtering
        
        # Apply filters if specified
        if filters:
            filtered_results = []
            for result in results:
                if self._matches_filters(result['metadata'], filters):
                    filtered_results.append(result)
            results = filtered_results[:k]
        else:
            results = results[:k]
        
        # Convert to SearchResult objects
        search_results = []
        for i, result in enumerate(results):
            search_results.append(SearchResult(
                text=result['text'],
                metadata=result['metadata'],
                score=1.0 - result['distance'],  # Convert distance to similarity score
                chunk_id=i
            ))
            
        return search_results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the specified filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
                
            if isinstance(value, list):
                # For list values, check intersection
                if not isinstance(metadata[key], list):
                    return False
                if not set(value) & set(metadata[key]):
                    return False
            elif isinstance(value, dict):
                # For dict values, all key-value pairs must match
                if not isinstance(metadata[key], dict):
                    return False
                if not all(metadata[key].get(k) == v for k, v in value.items()):
                    return False
            else:
                # For simple values, direct comparison
                if metadata[key] != value:
                    return False
                    
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded indices."""
        stats = {}
        for identifier, manager in self.embedding_managers.items():
            stats[identifier] = {
                'total_chunks': len(manager.texts),
                'total_tokens': sum(m.get('token_count', 0) for m in manager.metadata_list),
                'unique_files': len(set(m['file_path'] for m in manager.metadata_list))
            }
        return stats