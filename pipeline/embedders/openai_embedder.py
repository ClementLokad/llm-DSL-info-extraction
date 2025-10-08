"""
OpenAI embedder for code chunks using OpenAI's embedding API.

This embedder provides high-quality embeddings but requires API calls
and counts against OpenAI quotas.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import time

from pipeline.core.base_embedder import BaseEmbedder
from pipeline.core.base_chunker import CodeChunk

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI's embedding API for creating code embeddings.
    
    This embedder provides high-quality embeddings but requires API calls
    and uses quota. Good for when you need the best semantic understanding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Model configuration
        self.model_name = self.config.get('model_name', 'text-embedding-3-small')
        self.api_key = self.config.get('api_key', None)
        
        # Rate limiting
        self.requests_per_minute = self.config.get('requests_per_minute', 3000)
        self.tokens_per_minute = self.config.get('tokens_per_minute', 1000000)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # OpenAI client (initialized in initialize())
        self.client = None
        self._embedding_dim = None
        self._last_request_time = 0
        self._request_count = 0
        self._tokens_used = 0
        self._minute_start = time.time()
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        if self._embedding_dim is None:
            # Set based on model name
            model_dims = {
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
                'text-embedding-ada-002': 1536
            }
            self._embedding_dim = model_dims.get(self.model_name, 1536)
        
        return self._embedding_dim
    
    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai is required but not installed. "
                "Install it with: pip install openai"
            )
        
        try:
            # Get API key from config manager
            from config_manager import get_config
            config = get_config()
            api_key = self.api_key or config.get_api_key('OPENAI_API_KEY')
            
            if not api_key:
                raise RuntimeError(
                    "OpenAI API key not found. Set it in config or OPENAI_API_KEY environment variable"
                )
            
            self.client = openai.OpenAI(api_key=api_key)
            
            # Test the connection with a small request
            test_response = self.client.embeddings.create(
                model=self.model_name,
                input=["test"],
                encoding_format="float"
            )
            
            # Verify embedding dimension
            actual_dim = len(test_response.data[0].embedding)
            if self._embedding_dim != actual_dim:
                self.logger.warning(f"Expected embedding dim {self._embedding_dim}, got {actual_dim}")
                self._embedding_dim = actual_dim
            
            self._is_initialized = True
            self.logger.info(f"OpenAI embedder initialized. Model: {self.model_name}, Dimension: {self._embedding_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embedder: {e}")
    
    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Generate embeddings for a list of code chunks."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        if not chunks:
            return np.array([]).reshape(0, self.embedding_dimension)
        
        # Prepare texts for embedding
        texts = [self.prepare_chunk_for_embedding(chunk) for chunk in chunks]
        
        # Process in batches to respect rate limits
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_impl(batch)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        self.logger.debug(f"Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        # Prepare text
        prepared_text = self.prepare_text_for_embedding(text)
        
        # Generate embedding
        embedding = self._call_openai_api([prepared_text])
        
        return embedding[0]
    
    def _embed_batch_impl(self, texts: List[str]) -> np.ndarray:
        """Implementation-specific batch embedding method."""
        if not self._is_initialized:
            raise RuntimeError("Embedder must be initialized before use")
        
        # Prepare texts
        prepared_texts = [self.prepare_text_for_embedding(text) for text in texts]
        
        # Call OpenAI API
        embeddings = self._call_openai_api(prepared_texts)
        
        return embeddings
    
    def _call_openai_api(self, texts: List[str]) -> np.ndarray:
        """Call OpenAI embedding API with rate limiting and retry logic."""
        # Check rate limits
        self._check_rate_limits(texts)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float"
                )
                
                # Extract embeddings
                embeddings = []
                for item in response.data:
                    embeddings.append(item.embedding)
                
                # Update rate limiting counters
                self._update_rate_counters(texts)
                
                # Convert to numpy array
                embeddings_array = np.array(embeddings)
                
                # Normalize if requested
                if self.normalize:
                    embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                
                return embeddings_array
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"OpenAI API call failed after {self.max_retries} attempts: {e}")
    
    def _check_rate_limits(self, texts: List[str]) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._tokens_used = 0
            self._minute_start = current_time
        
        # Estimate tokens for current request
        estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough estimate
        
        # Check if we would exceed limits
        if (self._request_count + 1 > self.requests_per_minute or
            self._tokens_used + estimated_tokens > self.tokens_per_minute):
            
            # Wait until next minute
            wait_time = 60 - (current_time - self._minute_start)
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                # Reset counters
                self._request_count = 0
                self._tokens_used = 0
                self._minute_start = time.time()
    
    def _update_rate_counters(self, texts: List[str]) -> None:
        """Update rate limiting counters after successful API call."""
        self._request_count += 1
        # Update token count (rough estimate)
        estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)
        self._tokens_used += estimated_tokens
        self._last_request_time = time.time()
    
    def prepare_chunk_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare a code chunk for OpenAI embedding with optimized formatting."""
        text_parts = []
        
        # Add structured context for better understanding
        text_parts.append(f"Code Type: {chunk.chunk_type.replace('_', ' ').title()}")
        
        # Add semantic context
        if chunk.context:
            text_parts.append(f"Context: {chunk.context}")
        
        # Add metadata information
        if chunk.metadata:
            metadata_parts = []
            
            # Add chunk name if available
            if 'chunk_name' in chunk.metadata and chunk.metadata['chunk_name']:
                metadata_parts.append(f"name: {chunk.metadata['chunk_name']}")
            
            if 'section' in chunk.metadata:
                metadata_parts.append(f"section: {chunk.metadata['section']}")
            
            if chunk.chunk_type == 'data_ingestion' and 'table_names' in chunk.metadata:
                table_names = chunk.metadata['table_names']
                metadata_parts.append(f"tables: {', '.join(table_names)}")
            
            elif chunk.chunk_type == 'calculation' and 'variable_names' in chunk.metadata:
                from config_manager import get_config
                config = get_config()
                max_vars = config.get('embedder.text_preparation.max_variable_names', 3)
                var_names = chunk.metadata['variable_names'][:max_vars]
                metadata_parts.append(f"variables: {', '.join(var_names)}")
            
            if metadata_parts:
                text_parts.append(f"Metadata: {' | '.join(metadata_parts)}")
        
        # Add the main content with clear separation
        text_parts.append("Code:")
        text_parts.append(chunk.content)
        
        # Join with newlines and prepare
        full_text = '\n'.join(text_parts)
        return self.prepare_text_for_embedding(full_text)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        current_time = time.time()
        time_in_minute = current_time - self._minute_start
        
        return {
            "requests_this_minute": self._request_count,
            "tokens_this_minute": self._tokens_used,
            "requests_per_minute_limit": self.requests_per_minute,
            "tokens_per_minute_limit": self.tokens_per_minute,
            "time_in_current_minute": time_in_minute,
            "requests_remaining": max(0, self.requests_per_minute - self._request_count),
            "tokens_remaining": max(0, self.tokens_per_minute - self._tokens_used)
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get available OpenAI embedding models."""
        return {
            "text-embedding-3-small": {
                "dimension": 1536,
                "description": "Most capable small model, good performance",
                "cost_per_1m_tokens": 0.02,
                "max_input_tokens": 8191
            },
            "text-embedding-3-large": {
                "dimension": 3072,
                "description": "Most capable large model, best performance",
                "cost_per_1m_tokens": 0.13,
                "max_input_tokens": 8191
            },
            "text-embedding-ada-002": {
                "dimension": 1536,
                "description": "Legacy model, lower cost",
                "cost_per_1m_tokens": 0.10,
                "max_input_tokens": 8191
            }
        }
