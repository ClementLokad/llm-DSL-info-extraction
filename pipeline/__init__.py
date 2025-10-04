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

from preprocessing.core.base_parser import BaseParser
from preprocessing.core.base_chunker import BaseChunker  
from preprocessing.core.base_embedder import BaseEmbedder
from preprocessing.core.base_retriever import BaseRetriever

from preprocessing.parsers.envision_parser import EnvisionParser
from preprocessing.chunkers.semantic_chunker import SemanticChunker
from preprocessing.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from preprocessing.embedders.openai_embedder import OpenAIEmbedder
from preprocessing.embedders.gemini_embedder import GeminiEmbedder
from preprocessing.retrievers.faiss_retriever import FAISSRetriever

__version__ = "0.1.0"
__all__ = [
    "BaseParser", "BaseChunker", "BaseEmbedder", "BaseRetriever",
    "EnvisionParser", "SemanticChunker", 
    "SentenceTransformerEmbedder", "OpenAIEmbedder", "GeminiEmbedder",
    "FAISSRetriever"
]