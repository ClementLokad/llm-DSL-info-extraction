"""
Module for processing DSL scripts through the complete pipeline.
"""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from .chunker import DSLChunker, Chunk
from .tokenizer import TokenizerFactory
from .embeddings import EmbeddingManager, EmbeddingData
from .config import registry, ModelConfig

class DSLProcessor:
    """Handles the complete processing pipeline for DSL scripts."""
    
    def __init__(self, 
                 model_type: str, 
                 model_name: str,
                 max_chunk_size: int = 1000,
                 storage_dir: Optional[str] = None):
        """Initialize the DSL processor.
        
        Args:
            model_type: Type of model to use (e.g., 'gemini')
            model_name: Name of the specific model (e.g., 'gemini-pro')
            max_chunk_size: Maximum size of chunks
            storage_dir: Directory to store processed data
        """
        # Set up model configuration
        self.model_config = registry.get_config(model_type, model_name)
        self.chunker = DSLChunker(max_chunk_size=max_chunk_size)
        self.tokenizer = TokenizerFactory.get_tokenizer(model_type)
        
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir or 'processed_data'
        
        # Initialize embedding manager with model-specific configuration
        index_path = self.model_config.get_index_path(self.storage_dir)
        self.embedding_manager = EmbeddingManager(
            embedding_dim=self.model_config.embedding_dim,
            index_path=index_path
        )
        
    def process_file(self, file_path: str) -> List[Chunk]:
        """Process a single DSL file through the pipeline."""
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Chunk the content
        chunks = self.chunker.chunk_text(content)
        
        # Process chunks through the pipeline
        for chunk in chunks:
            # Add file information to metadata
            chunk.metadata['file_path'] = file_path
            chunk.metadata['file_name'] = os.path.basename(file_path)
            
            # Get token count
            chunk.metadata['token_count'] = self.tokenizer.get_token_count(chunk.text)
            
        return chunks
        
    def process_directory(self, dir_path: str, file_pattern: str = "*.nvn") -> Dict[str, List[Chunk]]:
        """Process all DSL files in a directory."""
        results = {}
        dir_path = Path(dir_path)
        
        for file_path in dir_path.glob(file_pattern):
            try:
                chunks = self.process_file(str(file_path))
                results[str(file_path)] = chunks
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        return results
        
    def generate_embeddings(self, chunks: List[Chunk], batch_size: int = 32) -> List[EmbeddingData]:
        """Generate embeddings for chunks using the appropriate model.
        
        Args:
            chunks: List of text chunks to process
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of EmbeddingData containing embeddings and metadata
        """
        from agents.base import LLMAgent
        from agents.gpt_agent import GPTAgent
        from agents.gemini_agent import GeminiAgent
        from agents.mistral_agent import MistralAgent
        import time
        
        # Get the appropriate agent class
        agent_classes = {
            'gemini': GeminiAgent,
            'gpt': GPTAgent,
            'mistral': MistralAgent
        }
        
        agent_class = agent_classes.get(self.model_config.model_type)
        if not agent_class:
            raise NotImplementedError(
                f"Embedding generation for model type '{self.model_config.model_type}' "
                "is not yet implemented"
            )
            
        # Initialize agent
        agent = agent_class()
        agent.initialize()
        
        # Generate embeddings
        embedding_data = []
        total_chunks = len(chunks)
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = []
            
            for chunk in batch:
                try:
                    embedding = agent.get_embedding(chunk.text)
                    embedding_data.append(EmbeddingData(
                        chunk=chunk,
                        embedding=embedding,
                        metadata=chunk.metadata
                    ))
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue
            
            # Report progress
            print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
            
        return embedding_data
        
    def save_processed_data(self):
        """Save all processed data to disk."""
        if self.storage_dir:
            self.embedding_manager.save()
            
    @classmethod
    def load_processor(cls, storage_dir: str, model_type: str, 
                      embedding_dim: int) -> 'DSLProcessor':
        """Load a previously saved processor state."""
        processor = cls(
            model_type=model_type,
            embedding_dim=embedding_dim,
            storage_dir=storage_dir
        )
        
        # Load the embedding manager
        index_path = os.path.join(storage_dir, f"{model_type}_index")
        if os.path.exists(index_path + '.index'):
            processor.embedding_manager = EmbeddingManager.load(index_path)
            
        return processor