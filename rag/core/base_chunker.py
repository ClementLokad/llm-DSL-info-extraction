"""
Base chunker interface for semantic code chunking.

This module defines the abstract interface for chunking code blocks into
semantically meaningful pieces optimized for embedding and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from rag.core.base_parser import CodeBlock

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """
    Represents a semantically meaningful chunk of code.
    
    Attributes:
        content: The chunked code content
        chunk_type: Type of chunk (semantic_block, function_group, etc.)
        original_blocks: List of original code blocks that form this chunk
        context: Surrounding context for better understanding
        size_tokens: Estimated token count (for embedding limits)
        metadata: Additional chunker-specific metadata
    """
    content: str
    chunk_type: str
    summary: str = None
    original_blocks: List[CodeBlock] = None
    context: str = ""
    size_tokens: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Estimate token count if not provided
        if self.size_tokens == 0:
            # Use configurable approximation ratio
            from config_manager import get_config
            chars_per_token = get_config().get('chunker.chars_per_token', 4)
            self.size_tokens = len(self.content) // chars_per_token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the CodeChunk to a dictionary for JSON serialization."""
        return {
            'content': self.content,
            'chunk_type': self.chunk_type,
            'original_blocks': [block.to_dict() for block in self.original_blocks],
            'context': self.context,
            'size_tokens': self.size_tokens,
            'metadata': self.metadata
        }

class BaseChunker(ABC):
    """
    Abstract base class for all code chunkers.
    
    This class defines the interface for chunking code blocks into semantically
    meaningful pieces suitable for embedding and retrieval.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Chunker-specific configuration options (if None, loads from global config)
                   Common options are loaded from config.yaml:
                   - max_chunk_tokens: Maximum tokens per chunk
                   - overlap_lines: Number of lines to overlap when splitting large blocks
                   - preserve_boundaries: Whether to preserve logical boundaries
        """
        from config_manager import get_config
        
        if config is None:
            # Load from global configuration
            global_config = get_config()
            self.config = global_config.get_chunker_config()
        else:
            # Use provided configuration
            self.config = config
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def max_chunk_tokens(self) -> int:
        """Maximum tokens per chunk (synchronized with config)."""
        return max(1, self.config.get('max_chunk_tokens', 512))
    
    @property
    def preserve_boundaries(self) -> bool:
        """Whether to preserve logical boundaries (synchronized with config)."""
        return self.config.get('preserve_boundaries', True)
    
    @property
    def chars_per_token(self) -> int:
        """Characters per token for estimation (synchronized with config)."""
        return max(1, self.config.get('chars_per_token', 4))
    
    @property
    def overlap_lines(self) -> int:
        """Number of lines to overlap when splitting large blocks (synchronized with config)."""
        return max(0, self.config.get('overlap_lines', 2))
    
    @abstractmethod
    def chunk_blocks(self, code_blocks: List[CodeBlock]) -> List[CodeChunk]:
        """
        Chunk a list of code blocks into semantic chunks.
        
        Args:
            code_blocks: List of parsed code blocks to chunk
            
        Returns:
            List of semantic code chunks
            
        Raises:
            ValueError: If input blocks are invalid
        """
        pass
    
    def chunk_single_block(self, code_block: CodeBlock) -> List[CodeChunk]:
        """
        Chunk a single large code block if it exceeds size limits.
        
        Args:
            code_block: Single code block to potentially split
            
        Returns:
            List of chunks (may be just one if block is small enough)
        """
        # Default implementation - can be overridden
        estimated_tokens = len(code_block.content) // self.chars_per_token
        
        if estimated_tokens <= self.max_chunk_tokens:
            # Block is small enough, return as single chunk
            return [CodeChunk(
                content=code_block.content,
                chunk_type="single_block",
                original_blocks=[code_block],
                size_tokens=estimated_tokens,
                metadata={
                    "source": "single_block",
                    "chunk_name": code_block.name or f"{code_block.block_type}_single"
                }
            )]
        else:
            # Block is too large, need to split
            return self._split_large_block(code_block)
    
    def _split_large_block(self, code_block: CodeBlock) -> List[CodeChunk]:
        """
        Split a large code block into smaller chunks.
        
        Args:
            code_block: Large code block to split
            
        Returns:
            List of smaller chunks
        """
        # Simple line-based splitting - can be overridden by specific chunkers
        lines = code_block.content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(line) // self.chars_per_token
            
            if current_tokens + line_tokens > self.max_chunk_tokens and current_chunk_lines:
                # Create chunk from accumulated lines
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    chunk_type="split_block",
                    original_blocks=[code_block],
                    size_tokens=current_tokens,
                    metadata={
                        "source": "split_block", 
                        "part": len(chunks) + 1,
                        "chunk_name": f"{code_block.name or code_block.block_type}_part_{len(chunks) + 1}"
                    }
                ))
                
                # Start new chunk with overlap based on configured line count
                if self.overlap_lines > 0 and len(current_chunk_lines) > self.overlap_lines:
                    overlap_line_list = current_chunk_lines[-self.overlap_lines:]  # Keep last N lines as overlap
                    current_chunk_lines = overlap_line_list + [line]
                    current_tokens = sum(len(l) // self.chars_per_token for l in overlap_line_list) + line_tokens
                else:
                    current_chunk_lines = [line]
                    current_tokens = line_tokens
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # Handle remaining lines
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(
                content=chunk_content,
                chunk_type="split_block",
                original_blocks=[code_block],
                size_tokens=current_tokens,
                metadata={
                    "source": "split_block", 
                    "part": len(chunks) + 1,
                    "chunk_name": f"{code_block.name or code_block.block_type}_part_{len(chunks) + 1}"
                }
            ))
        
        return chunks
    
    def validate_chunks(self, chunks: List[CodeChunk]) -> bool:
        """
        Validate that chunks meet size and quality requirements.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            True if all chunks are valid, False otherwise
        """
        for chunk in chunks:
            if chunk.size_tokens > self.max_chunk_tokens:
                self.logger.warning(f"Chunk exceeds max token limit: {chunk.size_tokens} > {self.max_chunk_tokens}")
                return False
            
            if not chunk.content.strip():
                self.logger.warning("Empty chunk found")
                return False
        
        return True
