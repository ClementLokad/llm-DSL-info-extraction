"""
Embedders for creating vector representations of code chunks.
"""

from pipeline.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

# Dynamic imports for embedders with optional dependencies
try:
    from pipeline.embedders.openai_embedder import OpenAIEmbedder
except ImportError:
    OpenAIEmbedder = None

try:
    from pipeline.embedders.gemini_embedder import GeminiEmbedder
except ImportError:
    GeminiEmbedder = None

__all__ = ["SentenceTransformerEmbedder"]
