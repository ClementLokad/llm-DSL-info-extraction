"""
Preprocessing pipeline for Envision DSL codebase analysis.

This module provides a modular preprocessing architecture for extracting,
chunking, embedding, and retrieving code segments from LOKAD's Envision DSL codebases.

The architecture supports:
- Flexible embedding backends (sentence-transformers, OpenAI, Gemini)
- Semantic code chunking respecting DSL structure
- Two-stage retrieval with LLM reranking
- Envision-specific code understanding
"""

from pipeline.core.base_parser import BaseParser
from pipeline.core.base_chunker import BaseChunker  
from pipeline.core.base_embedder import BaseEmbedder
from pipeline.core.base_retriever import BaseRetriever

from pipeline.parsers.envision_parser import EnvisionParser
from pipeline.chunkers.semantic_chunker import SemanticChunker
from pipeline.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from pipeline.retrievers.faiss_retriever import FAISSRetriever

# Dynamic imports for embedders with optional dependencies
try:
    from pipeline.embedders.openai_embedder import OpenAIEmbedder
except ImportError:
    OpenAIEmbedder = None

try:
    from pipeline.embedders.gemini_embedder import GeminiEmbedder
except ImportError:
    GeminiEmbedder = None

__version__ = "0.1.0"
__all__ = [
    "BaseParser", "BaseChunker", "BaseEmbedder", "BaseRetriever",
    "EnvisionParser", "SemanticChunker", 
    "SentenceTransformerEmbedder",
    "FAISSRetriever"
]
