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

from rag.parsers.old_envision_parser import EnvisionParser
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever

__version__ = "0.1.0"
__all__ = [
    "EnvisionParser", "EnvisionChunker",
    "SentenceTransformerEmbedder",
    "FAISSRetriever"
]
