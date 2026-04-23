from utils.config_manager import get_config

from rag.core.base_embedder import BaseEmbedder
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.embedders.gemini_embedder import GeminiEmbedder
from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.embedders.qdrant_embedder import QdrantEmbedder

from rag.core.base_retriever import BaseRetriever
from rag.retrievers.faiss_retriever import FAISSRetriever
from rag.retrievers.qdrant_retriever import QdrantRetriever

from rag.core.base_query_transformer import BaseQueryTransformer
from rag.query_transformers.fusion_query_transformer import FusionQueryTransformer 
from rag.query_transformers.hyde_query_transformer import HydeQueryTransformer 

def get_default_embedder(embedder_type: str = None) -> BaseEmbedder:
    if embedder_type is None:
        emb = get_config().get("embedder.default_type", "qdrant")
    else:
        emb = embedder_type.lower()
    
    if emb == "gemini":
        return GeminiEmbedder(get_config().get_embedder_config("gemini"))
    elif emb == "openai":
        return OpenAIEmbedder(get_config().get_embedder_config("openai"))
    elif emb == "sentence_transformer":
        return SentenceTransformerEmbedder(get_config().get_embedder_config("sentence_transformer"))
    elif emb == "qdrant":
        return QdrantEmbedder(get_config().get_embedder_config("qdrant"))
    else:
        raise ValueError(f"Unknown embedder type: {emb}")


def get_default_retriever(retriever_type: str = None) -> BaseRetriever:
    if retriever_type is None:
        ret = get_config().get("retriever.type", "qdrant")
    else:
        ret = retriever_type.lower()
    
    if ret == "faiss":
        return FAISSRetriever(get_config().get_retriever_config("faiss"))
    elif ret == "qdrant":
        return QdrantRetriever(get_config().get_retriever_config("qdrant"))
    else:
        raise ValueError(f"Unknown retriever type: {ret}")
    
def get_default_query_transformer(query_transformer_type: str = None) -> BaseQueryTransformer:
    if query_transformer_type is None:
        qt = get_config().get("query_transformer.query_transformer_type", "fusion")
    else:
        qt = query_transformer_type.lower()
    
    if not qt:
        return None
    elif qt == "hyde":
        return HydeQueryTransformer(get_config().get_query_transformer_config())
    elif qt == "fusion":
        return FusionQueryTransformer(get_config().get_query_transformer_config())
    else:
        raise ValueError(f"Unknown retriever type: {qt}")