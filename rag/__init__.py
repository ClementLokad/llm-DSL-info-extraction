"""
RAG (Retrieval-Augmented Generation) for Envision DSL codebase analysis.

This module provides a modular preprocessing architecture for extracting,
chunking, embedding, and retrieving code segments from LOKAD's Envision DSL codebases.

The architecture supports:
- Flexible embedding backends (sentence-transformers, OpenAI, Gemini)
- Semantic code chunking respecting DSL structure
- Two-stage retrieval with LLM reranking
- Envision-specific code understanding
"""

from rag.core.base_parser import BaseParser
from rag.core.base_chunker import BaseChunker  
from rag.core.base_embedder import BaseEmbedder
from rag.core.base_retriever import BaseRetriever

from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.summarizers.chunk_summarizer import ChunkSummarizer
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever

# Dynamic imports for embedders with optional dependencies
try:
    from rag.embedders.openai_embedder import OpenAIEmbedder
except ImportError:
    OpenAIEmbedder = None

try:
    from rag.embedders.gemini_embedder import GeminiEmbedder
except ImportError:
    GeminiEmbedder = None

__version__ = "0.1.0"
__all__ = [
    "BaseParser", "BaseChunker", "BaseEmbedder", "BaseRetriever",
    "EnvisionParser", "SemanticChunker", "ChunkSummarizer",
    "SentenceTransformerEmbedder",
    "FAISSRetriever"
]
