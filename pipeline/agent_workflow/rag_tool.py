from typing import List, Tuple
from pipeline.agent_workflow.workflow_base import BaseRAGTool
from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_embedder import BaseEmbedder
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
    
    def __init__(self, retriever: BaseRetriever, embedder: BaseEmbedder):
        super().__init__(retriever)
        self.embedder = embedder
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

        if get_config().get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + query)
            if verbose:
                print(f"Raw answer from LLM for decomposition of the query : {raw_questions}")
            questions = raw_questions.split("$")
            for sub_question in questions:
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
    
    def get_description(self) -> Tuple[str, str, List[str]]:
        usage = "Retrieve Envision concepts or Lokad business logic."
        parameter = (
            "A natural language query describing the concept to find.\n"
            "   Key words (Optionnal): add <key_words>KEYWORD1,KEYWORD2</key_words> to boost the relevance of results containing those keywords.\n"
            "   Sources (Optionnal): add <sources>PATH_REGEX</sources> to boost the relevance of results from files whose path matches the regex."
        )
        examples = [
            "<parameter>how does the refund policy work?</parameter>",
            "<parameter>how is growth defined? <key_words>growth</key_words></parameter>",
            "<parameter>What can we do on this account? <sources>/1. utilities/</sources></parameter>",
            "<parameter>To whom does this account belong? <sources>/7. Documentation/</sources><key_words>owner,belong</key_words></parameter>"
        ]
        
        return usage, parameter, examples

class AdvancedRAGTool(SimpleRAGTool):
    """
    An advanced implementation of BaseRAGTool.
    Thanks to the use of Qdrant's hybrid search capabilities, this tool can perform a single 
    search that combines both dense vector similarity and keyword-based sparse matching, with 
    configurable boosting and penalization based on keywords and source path patterns.
    
    This class uses a provided retriever to perform retrieval operations.
    """
    
    def __init__(self, retriever: QdrantRetriever, embedder: QdrantEmbedder):
        self.retriever = retriever
        self.embedder = embedder
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
        
        for i, result in enumerate(sorted_results, start=1):
            result.rank = i
            
        return sorted_results
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), verbose = False, key_words:List[str] = None, sources: List[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []
        
        retrieval_k = top_k
        if self.cross_encoding:
            retrieval_k = top_k*self.ce_multiplier

        if get_config().get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + query)
            if verbose:
                print(f"Raw answer from LLM for decomposition of the query : {raw_questions}")
            questions = raw_questions.split("$")
            for sub_question in questions:
                results.extend(self.retriever.search_hybrid(sub_question, self.embedder, top_k=retrieval_k,
                                                            keywords=key_words, source_substrings=sources))
            results = self.merge_rag_results(results)[:retrieval_k]
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
            results = self.reranker(enhanced_query, results)[:top_k]
        
        # Replace constants in result
        for result in results:
            cleaned_content = replace_constants_in_script(result.chunk.content, script_path=result.chunk.metadata["file_path"])
            result.chunk.content = cleaned_content

        return results
    
    def get_description(self) -> Tuple[str, str, List[str]]:
        usage = "Retrieve Envision concepts or Lokad business logic using SOTA Hybrid DB and cross-encoding."
        parameter = (
            "A natural language query describing the concept to find.\n"
            "   Key words (Optionnal): add <key_words>KEYWORD1,KEYWORD2</key_words> to boost the relevance of results containing those keywords.\n"
            "   Sources (Optionnal): add <sources>SUBSTRING1,SUBSTRING2</sources> to boost the relevance of results from files whose path contains at least one of the substrings."
        )
        examples = [
            "<parameter>how does the refund policy work?</parameter>",
            "<parameter>how is growth defined? <key_words>growth</key_words></parameter>",
            "<parameter>What can we do on this account? <sources>/1. utilities/,/7. Documentation/0 General Organization</sources></parameter>",
            "<parameter>To whom does this account belong? <sources>/7. Documentation/</sources><key_words>owner,belong</key_words></parameter>"
        ]
        
        return usage, parameter, examples

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