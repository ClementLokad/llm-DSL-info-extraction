from typing import List, Tuple, Dict, Any
from pipeline.agent_workflow.workflow_base import BaseRAGTool, _tool_desc
from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_embedder import BaseEmbedder
from rag.core.base_query_transformer import BaseQueryTransformer
from rag.retrievers.qdrant_retriever import QdrantRetriever
from rag.embedders.qdrant_embedder import QdrantEmbedder
from config_manager import get_config
from agents.prepare_agent import prepare_default_agent
from rag.utils.script_scanner import replace_constants_in_script
from pathlib import Path
from fastembed.rerank.cross_encoder import TextCrossEncoder
import time
import re
import math

class SimpleRAGTool(BaseRAGTool):
    """
    A simple implementation of BaseRAGTool.
    
    This class uses a provided retriever to perform retrieval operations.
    """
    
    def __init__(self, retriever: BaseRetriever, embedder: BaseEmbedder, query_transformer: BaseQueryTransformer = None):
        super().__init__(retriever)
        self.embedder = embedder
        self.query_transformer = query_transformer
        self.rate_limit_delay = get_config().get("agent.rate_limit_delay")
        self.agent = prepare_default_agent()
    
    def merge_rag_results(self, results):
        """Merges rag results from different sub-questions using Reciprocal Rank Fusion (RRF)"""
        k=10
        merged_results = {}
        for result in results:
            if result.chunk.content in merged_results.keys():
                score, chunk, metadata = merged_results[result.chunk.content]
                merged_results[result.chunk.content] = (score + 1/(k+result.rank), chunk, metadata)
                metadata.update(result.chunk.metadata)
            else:
                merged_results[result.chunk.content] = (1/(k+result.rank), result.chunk, result.metadata)
        results_list = sorted(merged_results.items(), key=lambda item: item[1][0], reverse = True)
        return [RetrievalResult(chunk, score, rank+1, metadata) for rank, (_, (score, chunk, metadata)) in enumerate(results_list)]
    
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), rerank_multiplier = get_config().get("rag.rerank_multiplier"), verbose = False, key_words:List[str] = None, sources: str = None) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []
        if key_words==None:
            key_words = []

        # Query transform mode: transform the query before retrieval
        if self.query_transformer:
            transformed_question_list = self.query_transformer.transform(query)
            
            for sub_question in transformed_question_list:
                emb = self.embedder.embed_text(sub_question)
                results.extend(self.retriever.search(emb, top_k=top_k*rerank_multiplier))
            results = self.merge_rag_results(results)[:top_k*rerank_multiplier]
        else:
            
            emb = self.embedder.embed_text(query)
            results = self.retriever.search(emb, top_k=top_k*rerank_multiplier)

        # Rerank results if needed
        if rerank_multiplier > 1:
            # Pre-compile regex outside the loop to save processing time
            source_pattern = re.compile(sources) if sources else None
            for result in results:
                chunk_text = result.chunk.content.lower()
                matches = sum(1 for key in key_words if key.lower() in chunk_text)
                if matches > 0:
                    result.score = min(1.0, result.score * (1 + 0.1* matches / len(key_words)))

                if source_pattern and not source_pattern.search(result.chunk.metadata.get("original_file_path", "")):
                    result.score = result.score * 0.85  # Penalize if source doesn't match
            # Sort results by score after reranking
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            for i, result in enumerate(sorted_results, start=1):
                result.rank = i
            results = sorted_results[:top_k]
        
        # Replace constants in result
        for result in results:
            cleaned_content = replace_constants_in_script(result.chunk.content, script_path=result.chunk.metadata["file_path"])
            result.chunk.content = cleaned_content

        return results
    
    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="rag_tool",
            description=(
                "Retrieve Envision concepts or Lokad business logic using semantic search."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": (
                        "A natural language query describing the concept to find. "
                        "E.g. 'how does the refund policy work?' or 'how is growth defined?'."
                    ),
                },
                "key_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of exact keywords to boost the relevance of results "
                        "containing those keywords. "
                        "E.g. ['growth'] or ['owner', 'belong']."
                    ),
                },
                "sources": {
                    "type": "string",
                    "description": (
                        "Optional path regex to boost the relevance of results from files "
                        "whose path matches. "
                        "E.g. '/1. utilities/' or '/7. Documentation/0 General Organization'."
                    ),
                },
            },
            required=["query"],
        )

class AdvancedRAGTool(SimpleRAGTool):
    """
    An advanced implementation of BaseRAGTool.
    Thanks to the use of Qdrant's hybrid search capabilities, this tool can perform a single 
    search that combines both dense vector similarity and keyword-based sparse matching, with 
    configurable boosting and penalization based on keywords and source path patterns.
    
    This class uses a provided retriever to perform retrieval operations.
    """
    
    def __init__(self, retriever: QdrantRetriever, embedder: QdrantEmbedder, query_transformer):
        self.retriever = retriever
        self.embedder = embedder
        self.query_transformer = query_transformer
        self.rate_limit_delay = get_config().get("agent.rate_limit_delay")
        self.cross_encoding = get_config().get("main_pipeline.rag_tool.cross_encoding", False)
        self.ce_multiplier = get_config().get("main_pipeline.rag_tool.cross_encoding_multiplier", 3)
        self.agent = prepare_default_agent()
        self.reranker_model = TextCrossEncoder(get_config().get("main_pipeline.rag_tool.cross_encoder",
                                                              "jinaai/jina-reranker-v2-base-multilingual"))

    def ce_reranker(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank the retrieved results using fastembed cross-encoder"""
        if not results: return []
        
        # Prepare the documents to pair with the query
        documents = [f"Source: {result.chunk.metadata['original_file_path']}\n" + result.chunk.content for result in results]
        
        # FastEmbed handles batching, truncation, and output parsing automatically!
        # It yields a generator of scores (floats)
        scores = list(self.reranker_model.rerank(query, documents))
        
        # Apply Sigmoid to convert logits to 0-1 probabilities
        for result, logit in zip(results, scores):
            # Safe sigmoid calculation
            probability = 1 / (1 + math.exp(-logit))
            result.score = probability
            
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # 2. Dynamic Threshold Logic
        final_results = []
        best_score = sorted_results[0].score
        
        # Define your drop-off tolerance (e.g., anything worse than 50% of the best score is dropped)
        # Note: You can expose this in your config: get_config().get("rag.dynamic_threshold_ratio", 0.5)
        drop_off_ratio = get_config().get("main_pipeline.rag_tool.drop_off_ratio", 0.5)
        
        # Optional: Define an absolute minimum so we don't pass absolute garbage if the "best" score is terrible
        absolute_minimum = get_config().get("main_pipeline.rag_tool.absolute_minimum", 0.2)
        
        for result in sorted_results:
            # If the score is less than the absolute minimum, stop completely
            if result.score < absolute_minimum:
                break
            
            # If the score drops significantly below the top performer, cut the list off here
            if result.score < (best_score * drop_off_ratio):
                # We found the "gap" where relevance falls off a cliff
                break 
                
            final_results.append(result)
        
        # Assign ranks only to the surviving chunks
        for i, result in enumerate(final_results, start=1):
            result.rank = i
            
        return final_results
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), verbose = False, key_words:List[str] = None, sources: List[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []
        
        retrieval_k = top_k
        if self.cross_encoding:
            retrieval_k = top_k*self.ce_multiplier

        # Query transform mode: transform the query before retrieval
        if self.query_transformer:
            transformed_question_list = self.query_transformer.transform(query)

            for sub_question in transformed_question_list:
                results.extend(self.retriever.search_hybrid(sub_question, self.embedder, top_k=retrieval_k,
                                                            keywords=key_words, source_substrings=sources))
            results = self.merge_rag_results(results)[:retrieval_k*2]
        else:
            results = self.retriever.search_hybrid(query, self.embedder, top_k=retrieval_k,
                                                       keywords=key_words, source_substrings=sources)
        
        if self.cross_encoding:
            # 1. Build an enhanced query that a Transformer can understand contextually
            enhanced_query = query
            
            context_parts = []
            if key_words:
                context_parts.append(f"Important keywords: {', '.join(key_words)}")
            if sources:
                context_parts.append(f"Sources of interest: {', '.join(sources)}")
                
            # If we have extra context, append it cleanly inside parentheses
            if context_parts:
                enhanced_query += f" ({'; '.join(context_parts)})"
                
            # 2. Pass the enhanced query to the reranker
            results = self.ce_reranker(enhanced_query, results)[:top_k]
        
        # Replace constants in result
        for result in results:
            cleaned_content = replace_constants_in_script(result.chunk.content, script_path=result.chunk.metadata["file_path"])
            result.chunk.content = cleaned_content

        return results
    
    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="rag_tool",
            description=(
                "Semantic search over the knowledge base using SOTA Hybrid DB and cross-encoding. "
                "Combines dense vector similarity and keyword-based sparse matching."
                "This is the DEFAULT starting tool for most questions. "
                "Use whenever the question involves a concept, a behaviour, business logic, "
                "or any term that could appear in many different contexts. "
                "Also use when a prior grep returned too many results or irrelevant matches — "
                "semantic search will find the most relevant chunks regardless of exact wording."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": (
                        "A natural language query describing the concept to find. "
                        "E.g. 'how does the refund policy work?' or 'how is growth defined?'."
                    ),
                },
                "key_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of exact keywords to boost the relevance of results "
                        "containing those keywords. "
                        "E.g. ['growth'] or ['owner', 'belong']."
                    ),
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of path substrings to boost the relevance of results "
                        "from files whose path contains at least one of the substrings. "
                        "E.g. ['/1. utilities/', '/7. Documentation/0 General Organization']."
                    ),
                },
            },
            required=["query"],
        )

if __name__ == "__main__":
    from rag.retrievers.qdrant_retriever import QdrantRetriever
    from rag.embedders.qdrant_embedder import QdrantEmbedder
    
    embedder = QdrantEmbedder(get_config().get_embedder_config())
    embedder.initialize()
    retriever = QdrantRetriever(get_config().get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    # Determine index type from flags and config
    index_type = get_config().get("embedder.index_type", "full_chunk")
    if index_type == "full_chunk":
        index_path = Path("data/faiss_index")
    if index_type == "summary":
        index_path = Path("data/faiss_summary_index")
    
    retriever.load_index(str(index_path))
    
    rag_tool = AdvancedRAGTool(retriever, embedder)
    query = "How is the cost of possessing (holding) a unit of product in stock modeled in Envision?"
    try:
        results = rag_tool.retrieve(query, verbose=True, key_words=["cost", "possessing", "holding"], sources=["/7. Documentation/", "/4. Optimization workflow"])
    finally:
        retriever.close()

    for result in results:
        print("\n\n" + "="*60 + "\n\n")
        print(f"Score: {result.score*100:.2f}%, Rank: {result.rank}, Path: {result.chunk.metadata['original_file_path']}\nContent: {result.chunk.content}\n")