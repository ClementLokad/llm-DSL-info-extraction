"""
FAISS-based retriever for fast similarity search on code embeddings.

This retriever uses Facebook's FAISS library for efficient vector similarity
search, supporting both CPU and GPU acceleration.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import pickle
import os

from preprocessing.core.base_retriever import BaseRetriever, RetrievalResult
from preprocessing.core.base_chunker import CodeChunk

logger = logging.getLogger(__name__)

class FAISSRetriever(BaseRetriever):
    """
    FAISS-based retriever for fast similarity search on code embeddings.
    
    This retriever provides efficient similarity search using FAISS,
    supporting various index types and optimization strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # FAISS configuration - merge global and faiss-specific config
        faiss_config = self.config.get('faiss', {})
        
        # Index type - prioritize faiss-specific, fallback to global, then default
        self.index_type = (faiss_config.get('index_type') or 
                          faiss_config.get('default_index_type') or 
                          self.config.get('index_type', 'IndexFlatIP'))
        
        # GPU configuration
        self.use_gpu = faiss_config.get('use_gpu', False)
        
        # IVF parameters
        self.nlist = faiss_config.get('nlist', 100)  # For IVF indices
        self.nprobe = faiss_config.get('nprobe', 10)  # For IVF indices
        
        # HNSW parameters
        self.hnsw_m = faiss_config.get('m', 16)  # Number of connections
        self.hnsw_ef_construction = faiss_config.get('ef_construction', 200)  # Build time parameter
        self.hnsw_ef_search = faiss_config.get('ef_search', 64)  # Search time parameter
        
        # File names from configuration (NO hardcoded constants!)
        files_config = self.config.get('files', {})
        paths_config = self.config.get('paths', {})
        
        # Construct filenames with proper extensions from global config
        pickle_ext = paths_config.get('pickle_extension', '.pkl')
        index_ext = paths_config.get('index_extension', '.index')
        
        chunks_base = files_config.get('chunks_filename', 'chunks')
        index_base = files_config.get('index_filename', 'faiss')
        metadata_base = files_config.get('metadata_filename', 'metadata')
        
        self._chunks_filename = f"{chunks_base}{pickle_ext}"
        self._index_filename = f"{index_base}{index_ext}"
        self._metadata_filename = f"{metadata_base}{pickle_ext}"
        
        # FAISS objects (initialized in initialize())
        self.faiss = None
        self.index = None
        self._embedding_dim = None
    
    def _get_chunks_file_path(self, base_path: Optional[str] = None) -> str:
        """Get the full path to the chunks pickle file."""
        return os.path.join(base_path or self.index_path, self._chunks_filename)
    
    def _get_index_file_path(self, base_path: Optional[str] = None) -> str:
        """Get the full path to the FAISS index file."""
        return os.path.join(base_path or self.index_path, self._index_filename)
    
    def _get_metadata_file_path(self, base_path: Optional[str] = None) -> str:
        """Get the full path to the metadata pickle file."""
        return os.path.join(base_path or self.index_path, self._metadata_filename)
    
    def initialize(self, embedding_dimension: int) -> None:
        """Initialize the FAISS retriever with the specified embedding dimension."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "faiss-cpu is required but not installed. "
                "Install it with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        try:
            self.faiss = faiss
            self._embedding_dim = embedding_dimension
            
            # Create index based on configuration
            self.index = self._create_index(embedding_dimension)
            
            # Create index directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            self._is_initialized = True
            self.logger.info(f"FAISS retriever initialized. Index type: {self.index_type}, Dimension: {embedding_dimension}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS retriever: {e}")
    
    def _create_index(self, dimension: int):
        """Create a FAISS index based on configuration."""
        if self.index_type == 'IndexFlatIP':
            # Flat index with inner product (best for cosine similarity with normalized vectors)
            index = self.faiss.IndexFlatIP(dimension)
        
        elif self.index_type == 'IndexFlatL2':
            # Flat index with L2 distance
            index = self.faiss.IndexFlatL2(dimension)
        
        elif self.index_type == 'IndexIVFFlat':
            # IVF (Inverted File) index for faster search with some accuracy loss
            quantizer = self.faiss.IndexFlatIP(dimension)
            index = self.faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
        
        elif self.index_type == 'IndexHNSWFlat':
            # HNSW (Hierarchical Navigable Small World) index
            index = self.faiss.IndexHNSWFlat(dimension, self.hnsw_m)
            # Set construction parameter
            index.hnsw.efConstruction = self.hnsw_ef_construction
            # Set search parameter (can be changed later via optimize_index)
            index.hnsw.efSearch = self.hnsw_ef_search
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Move to GPU if requested and available
        if self.use_gpu and hasattr(self.faiss, 'StandardGpuResources'):
            try:
                res = self.faiss.StandardGpuResources()
                index = self.faiss.index_cpu_to_gpu(res, 0, index)
                self.logger.info("Using GPU acceleration for FAISS index")
            except Exception as e:
                self.logger.warning(f"Failed to use GPU, falling back to CPU: {e}")
        
        return index
    
    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """Add code chunks and their embeddings to the retriever."""
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")
        
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Number of chunks ({len(chunks)}) doesn't match embeddings ({embeddings.shape[0]})")
        
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) doesn't match expected ({self._embedding_dim})")
        
        # Ensure embeddings are in the right format (float32)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings if using inner product index
        if self.index_type in ['IndexFlatIP', 'IndexIVFFlat']:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Train index if necessary (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.logger.info("Training FAISS index...")
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunks
        self._chunks.extend(chunks)
        
        self.logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.get_chunk_count()}")
    
    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Search for similar code chunks using query embedding."""
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")
        
        if self.get_chunk_count() == 0:
            return []
        
        top_k = top_k or self.default_top_k
        top_k = min(top_k, self.get_chunk_count())  # Can't return more than we have
        
        # Ensure query embedding is in the right format
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding if using inner product index
        if self.index_type in ['IndexFlatIP', 'IndexIVFFlat']:
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Set search parameters for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Perform search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Convert results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # -1 means no more results
                chunk = self._chunks[idx]
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=i + 1,
                    metadata={
                        'faiss_index': int(idx),
                        'similarity_metric': self.similarity_metric
                    }
                )
                results.append(result)
        
        return results
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save the current index to disk."""
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before saving")
        
        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        index_file = self._get_index_file_path(save_path)
        
        # Move index to CPU if it's on GPU for saving
        index_to_save = self.index
        if hasattr(self.faiss, 'index_gpu_to_cpu') and self.use_gpu:
            try:
                index_to_save = self.faiss.index_gpu_to_cpu(self.index)
            except:
                pass  # If it fails, the index might already be on CPU
        
        self.faiss.write_index(index_to_save, index_file)
        
        # Save chunks
        chunks_file = self._get_chunks_file_path(save_path)
        with open(chunks_file, 'wb') as f:
            pickle.dump(self._chunks, f)
        
        # Save metadata
        metadata = {
            'embedding_dimension': self._embedding_dim,
            'index_type': self.index_type,
            'chunk_count': len(self._chunks),
            'use_gpu': self.use_gpu,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'hnsw_m': self.hnsw_m,
            'hnsw_ef_construction': self.hnsw_ef_construction,
            'hnsw_ef_search': self.hnsw_ef_search
        }
        
        metadata_file = self._get_metadata_file_path(save_path)
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Saved FAISS index with {len(self._chunks)} chunks to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load an index from disk."""
        load_path = path or self.index_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index path does not exist: {load_path}")
        
        # Load metadata
        metadata_file = self._get_metadata_file_path(load_path)
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Initialize if not already done
        if not self._is_initialized:
            self.initialize(metadata['embedding_dimension'])
        
        # Load FAISS index
        index_file = self._get_index_file_path(load_path)
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        
        self.index = self.faiss.read_index(index_file)
        
        # Restore HNSW parameters if applicable
        if self.index_type == 'IndexHNSWFlat' and hasattr(self.index, 'hnsw'):
            # Restore efSearch parameter (efConstruction is only used during building)
            if 'hnsw_ef_search' in metadata:
                self.index.hnsw.efSearch = metadata['hnsw_ef_search']
        
        # Move to GPU if requested
        if self.use_gpu and hasattr(self.faiss, 'StandardGpuResources'):
            try:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                self.logger.warning(f"Failed to move loaded index to GPU: {e}")
        
        # Load chunks
        chunks_file = self._get_chunks_file_path(load_path)
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'rb') as f:
            self._chunks = pickle.load(f)
        
        self.logger.info(f"Loaded FAISS index with {len(self._chunks)} chunks from {load_path}")
    
    def clear_index(self) -> None:
        """Clear all stored chunks and embeddings."""
        super().clear_index()
        
        if self.index is not None:
            # Reset the index
            self.index.reset()
        
        self.logger.info("Cleared FAISS index")
    
    def optimize_index(self) -> bool:
        """
        Optimize the index for better performance.
        
        Returns:
            bool: True if optimization was performed, False otherwise
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before optimization")
        
        if self.get_chunk_count() == 0:
            self.logger.warning("Cannot optimize empty index")
            return False
        
        # For HNSW indices, we can optimize search parameters
        if self.index_type == 'IndexHNSWFlat' and hasattr(self.index, 'hnsw'):
            # Optimize ef_search parameter based on data size
            data_size = self.index.ntotal
            if data_size > 10000:
                self.index.hnsw.efSearch = 128  # Higher for larger datasets
            elif data_size > 1000:
                self.index.hnsw.efSearch = 64   # Medium for medium datasets
            else:
                self.index.hnsw.efSearch = 32   # Lower for small datasets
            
            self.logger.info(f"Optimized HNSW efSearch parameter to {self.index.hnsw.efSearch} for {data_size} vectors")
            return True
        
        # For IVF indices, ensure they are properly trained
        if hasattr(self.index, 'is_trained') and not self.index.is_trained and self.index.ntotal > 0:
            # This shouldn't happen in normal usage, but just in case
            self.logger.warning("IVF index is not trained, this may indicate an issue")
            return False
        
        # For flat indices, no optimization is needed
        if self.index_type in ['IndexFlatIP', 'IndexFlatL2']:
            self.logger.info("No optimization needed for flat index types")
            return False
        
        self.logger.info("Index optimization check completed")
        return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "index_type": self.index_type,
            "embedding_dimension": self._embedding_dim,
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks": len(self._chunks),
            "use_gpu": self.use_gpu,
            "similarity_metric": self.similarity_metric
        }
        
        # Add index-specific information
        if hasattr(self.index, 'is_trained'):
            info["is_trained"] = self.index.is_trained
        
        if hasattr(self.index, 'nlist'):
            info["nlist"] = self.nlist
            info["nprobe"] = self.nprobe
        
        return info
    
    @classmethod
    def get_recommended_index_types(cls) -> Dict[str, Dict[str, Any]]:
        """Get recommended index types for different use cases."""
        return {
            "exact_search": {
                "index_type": "IndexFlatIP",
                "description": "Exact search, best quality but slower for large datasets",
                "use_case": "High accuracy required, small to medium datasets"
            },
            "fast_search": {
                "index_type": "IndexIVFFlat",
                "description": "Approximate search, good balance of speed and accuracy",
                "use_case": "Large datasets, can tolerate slight accuracy loss",
                "config": {"nlist": 100, "nprobe": 10}
            },
            "memory_efficient": {
                "index_type": "IndexHNSWFlat",
                "description": "Memory efficient with good performance",
                "use_case": "Memory constrained environments"
            }
        }