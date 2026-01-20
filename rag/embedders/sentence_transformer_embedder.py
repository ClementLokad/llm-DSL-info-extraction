"""
Sentence-transformers based embedder for code chunks.

This embedder uses sentence-transformers models to create embeddings
without relying on LLM APIs, making it suitable for high-volume processing.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from rag.core.base_embedder import BaseEmbedder
from rag.core.base_chunker import CodeChunk

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder using sentence-transformers for creating code embeddings.
    
    This embedder is ideal for processing large volumes of code without
    hitting API quotas, while still providing good semantic understanding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model configuration
        self.model_name = self.config.get('embedder.sentence_transformer.model_name', 'all-MiniLM-L6-v2')
        self.model_path = self.config.get('embedder.sentence_transformer.model_path', "data/sentence_transformer")  # Local path if available
        self.device = self.config.get('device', None)  # None = auto-detect
        self.trust_remote_code = self.config.get('trust_remote_code', False)
        
        # Processing configuration
        self.show_progress_bar = self.config.get('show_progress_bar', True)
        self.convert_to_numpy = self.config.get('convert_to_numpy', True)
        
        # Model instance (initialized in initialize())
        self.model = None
        self._embedding_dim = None
        
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        if self._embedding_dim is None:
            if not self._is_initialized:
                raise RuntimeError("Embedder must be initialized to get embedding dimension")
            # Get dimension from model
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        
        return self._embedding_dim
    
    def initialize(self) -> None:
        """Initialize the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required but not installed. "
                "Install it with: pip install sentence-transformers"
            )
        
        try:
            self.logger.info(f"Loading sentence-transformer model: {self.model_name}")
            
            try:
                self.model = SentenceTransformer(self.model_path,
                                                 device=self.device,
                                                 trust_remote_code=self.trust_remote_code, 
                                                 local_files_only=True)
            except Exception: 
                print(f"Failed to load model from path {self.model_path}, saving model_name from hub.")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code
                )
                self.model.save(self.model_path)  # Save locally for future use
            
            # Cache embedding dimension
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self._is_initialized = True
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize sentence-transformer model: {e}")
    
    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Generate embeddings for a list of code chunks."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        if not chunks:
            return np.array([]).reshape(0, self.embedding_dimension)

        # Prepare chunks for embedding
        texts = [self.prepare_chunk_for_embedding(chunk) for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=self.convert_to_numpy,
            normalize_embeddings=self.normalize
        )
        
        # Ensure we have a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        self.logger.debug(f"Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        # Prepare text
        prepared_text = self.prepare_text_for_embedding(text)
        
        # Generate embedding
        embedding = self.model.encode(
            [prepared_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=self.convert_to_numpy,
            normalize_embeddings=self.normalize
        )
        
        # Ensure we have a numpy array and correct shape
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        if len(embedding.shape) == 2:
            embedding = embedding[0]  # Remove batch dimension
        
        return embedding
    
    def _embed_batch_impl(self, texts: List[str]) -> np.ndarray:
        """Implementation-specific batch embedding method."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        # Prepare texts for embedding
        prepared_texts = [self.prepare_text_for_embedding(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=len(prepared_texts),  # Process entire batch at once
            show_progress_bar=False,
            convert_to_numpy=self.convert_to_numpy,
            normalize_embeddings=self.normalize
        )
        
        # Ensure we have a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def prepare_chunk_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare a code chunk for embedding with code-specific formatting."""

        # Use summary from metadata if available
        if "summary" in chunk.metadata and chunk.metadata["summary"]:
            return self.prepare_text_for_embedding(chunk.metadata["summary"])
        
        text_parts = []
        
        # Add section information
        if 'section' in chunk.metadata:
            text_parts.append(f"Section: {chunk.metadata['section']}")
        
        # Add the main content with code formatting hints
        
        text_parts.append(chunk.content)
        
        # Join with newlines and prepare
        full_text = '\n'.join(text_parts)
        return self.prepare_text_for_embedding(full_text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "status": "initialized"
        }
        
        # Add tokenizer info if available
        if hasattr(self.model, 'tokenizer'):
            info["tokenizer"] = self.model.tokenizer.__class__.__name__
        
        return info
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get recommended models for different use cases."""
        return {
            "general_code": {
                "model_name": "all-MiniLM-L6-v2",
                "description": "Fast and efficient for general code understanding",
                "embedding_dim": 384,
                "performance": "good",
                "speed": "fast"
            },
            "multilingual": {
                "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Good for mixed language codebases",
                "embedding_dim": 384,
                "performance": "good",
                "speed": "medium"
            },
            "high_quality": {
                "model_name": "all-mpnet-base-v2",
                "description": "Higher quality embeddings, slower processing",
                "embedding_dim": 768,
                "performance": "excellent",
                "speed": "slow"
            },
            "code_specific": {
                "model_name": "microsoft/codebert-base",
                "description": "Specialized for code understanding",
                "embedding_dim": 768,
                "performance": "excellent_for_code",
                "speed": "slow",
                "note": "Requires additional setup"
            }
        }
    
    def benchmark_performance(self, sample_texts: List[str], num_runs: int = 3) -> Dict[str, float]:
        """Benchmark embedding performance on sample texts."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before benchmarking")
        
        import time
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.embed_batch(sample_texts)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        texts_per_second = len(sample_texts) / avg_time
        
        return {
            "average_time_seconds": avg_time,
            "texts_per_second": texts_per_second,
            "num_texts": len(sample_texts),
            "num_runs": num_runs
        }
