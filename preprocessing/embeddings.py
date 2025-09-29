"""
Module for generating and managing embeddings.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

@dataclass
class EmbeddingData:
    """Container for embedding data and metadata."""
    embedding: np.ndarray
    metadata: Dict[str, Any]
    text: str

class EmbeddingManager:
    """Manages embedding generation and storage using FAISS."""
    
    def __init__(self, embedding_dim: int, index_path: Optional[str] = None):
        """Initialize the embedding manager.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load FAISS index
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata_list: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        
    def add_embeddings(self, embeddings: List[EmbeddingData]):
        """Add embeddings to the index."""
        if not embeddings:
            return
            
        vectors = np.vstack([e.embedding for e in embeddings])
        self.index.add(vectors)
        self.metadata_list.extend([e.metadata for e in embeddings])
        self.texts.extend([e.text for e in embeddings])
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in the index.
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                result = {
                    'text': self.texts[idx],
                    'metadata': self.metadata_list[idx],
                    'distance': float(dist),
                    'rank': i + 1
                }
                results.append(result)
                
        return results
        
    def save(self, path: Optional[str] = None):
        """Save the index and metadata to disk."""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path specified")
            
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.index')))
        
        # Save metadata and texts
        metadata_path = path.with_suffix('.metadata')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata_list': self.metadata_list,
                'texts': self.texts,
                'embedding_dim': self.embedding_dim
            }, f)
            
    @classmethod
    def load(cls, path: str) -> 'EmbeddingManager':
        """Load an embedding manager from disk."""
        path = Path(path)
        
        # Load metadata and texts
        metadata_path = path.with_suffix('.metadata')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            
        # Create instance
        instance = cls(embedding_dim=data['embedding_dim'], index_path=str(path))
        instance.metadata_list = data['metadata_list']
        instance.texts = data['texts']
        
        # Load FAISS index
        instance.index = faiss.read_index(str(path.with_suffix('.index')))
        
        return instance