import numpy as np
from typing import List, Dict, Any, Optional

from fastembed import TextEmbedding, SparseTextEmbedding

from rag.core.base_embedder import BaseEmbedder
from rag.core.base_chunker import CodeChunk

class QdrantEmbedder(BaseEmbedder):
    """
    Hybrid embedder using fastembed for creating dense and sparse code embeddings.
    
    This embedder satisfies the BaseEmbedder interface by returning dense semantic 
    vectors (based on LLM summaries) for standard methods, while providing dedicated 
    hybrid methods to return both dense and sparse (BM25) vectors for Qdrant insertion.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Fixed the typo here to grab the correct dense model fallback
        self.dense_model_name = self.config.get("dense_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model_name = self.config.get("sparse_model_name", "Qdrant/bm25")
        self.disable_stemmer = self.config.get("disable_stemmer", True)
        
        # Default dimension for all-MiniLM-L6-v2 if not specified
        self._embedding_dimension = self.config.get("embedding_dimension", 384)

        self.dense_model = None
        self.sparse_model = None

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def initialize(self) -> None:
        """
        Initializes both the dense and sparse fastembed models via ONNX runtime.
        """
        self.logger.info(f"Initializing Dense Model: {self.dense_model_name}")
        self.dense_model = TextEmbedding(model_name=self.dense_model_name)
        
        self.logger.info(f"Initializing Sparse Model: {self.sparse_model_name} (disable_stemmer={self.disable_stemmer})")
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name, disable_stemmer=self.disable_stemmer)
        
        self._is_initialized = True
        self.logger.info("QdrantEmbedder initialized successfully.")

    def _embed_batch_impl(self, texts: List[str]) -> np.ndarray:
        """
        Implementation for the base class batch embedding.
        Returns ONLY dense embeddings to satisfy the np.ndarray return type.
        """
        # fastembed returns a generator, so we cast to list and then to numpy array
        embeddings = list(self.dense_model.embed(texts))
        return np.array(embeddings)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a dense embedding for a single text string.
        """
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
            
        prepared_text = self.prepare_text_for_embedding(text)
        return self._embed_batch_impl([prepared_text])[0]

    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """
        Generate dense embeddings for a list of chunks (uses summaries if available).
        """
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
            
        texts = [self.prepare_chunk_for_embedding(chunk) for chunk in chunks]
        return self.embed_batch(texts)

    # ---------------------------------------------------------
    # QDRANT-SPECIFIC HYBRID METHODS
    # ---------------------------------------------------------

    def embed_text_hybrid(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Generate both dense and sparse embeddings for a query.
        
        Args:
            text: The user's original search query.
            
        Returns:
            Dictionary containing 'dense' (list of floats) and 'sparse' (SparseEmbedding object).
        """
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")

        prepared_text = self.prepare_text_for_embedding(text)
        
        dense_vec = list(self.dense_model.embed([prepared_text]))[0]
        # Sparse matching shouldn't be heavily truncated/cleaned so it catches exact syntax
        sparse_vec = list(self.sparse_model.embed([text+" "+" ".join(keywords)]))[0] 
        
        return {
            "dense": dense_vec.tolist(),
            "sparse": sparse_vec
        }

    def embed_chunks_hybrid(self, chunks: List[CodeChunk]) -> List[Dict[str, Any]]:
        """
        Generate both dense (from summary) and sparse (from raw code) embeddings.
        This is the method you should call before inserting into Qdrant.
        
        Args:
            chunks: List of CodeChunks.
            
        Returns:
            List of dictionaries containing dense and sparse vectors for each chunk.
        """
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")

        # 1. Dense relies on your prepared text (which gracefully uses LLM summaries)
        dense_texts = [self.prepare_chunk_for_embedding(c) for c in chunks]
        
        # 2. Sparse relies on the raw code content to catch exact variable names and syntax
        sparse_texts = [c.content for c in chunks]
        
        # Process both in batches natively via fastembed
        dense_vecs = list(self.dense_model.embed(dense_texts, batch_size=self.batch_size))
        sparse_vecs = list(self.sparse_model.embed(sparse_texts, batch_size=self.batch_size))
        
        results = []
        for d_vec, s_vec in zip(dense_vecs, sparse_vecs):
            results.append({
                "dense": d_vec.tolist(),
                "sparse": s_vec  # Qdrant expects this as an object/dict depending on the client version
            })
            
        return results