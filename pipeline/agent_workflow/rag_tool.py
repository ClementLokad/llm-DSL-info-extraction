from typing import List
from pipeline.agent_workflow.workflow_base import BaseRAGTool
from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_embedder import BaseEmbedder
from config_manager import get_config
from agents.prepare_agent import prepare_default_agent
from rag.utils.script_scanner import replace_constants_in_script
from pathlib import Path
import time
import re

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
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), rerank_multiplier = get_config().get("rag.rerank_multiplier"), verbose = False, key_words:List[str] = None, sources_regex: str = None) -> List[RetrievalResult]:
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
            source_pattern = re.compile(sources_regex) if sources_regex else None
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

if __name__ == "__main__":
    from rag.retrievers.faiss_retriever import FAISSRetriever
    from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(get_config().get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(get_config().get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    # Determine index type from flags and config
    index_type = get_config().get("embedder.index_type", "full_chunk")
    if index_type == "full_chunk":
        index_path = Path("data/faiss_index")
    if index_type == "summary":
        index_path = Path("data/faiss_summary_index")
    
    retriever.load_index(str(index_path))
    
    rag_tool = SimpleRAGTool(retriever, embedder)
    query = "How is the cost of possessing (holding) a unit of product in stock modeled in Envision?"
    results = rag_tool.retrieve(query, verbose=True, key_words=["cost", "possessing", "holding"])
    
    for result in results:
        print("\n\n" + "="*60 + "\n\n")
        print(f"Score: {result.score}, Rank: {result.rank}, Path: {result.chunk.metadata['original_file_path']}\nContent: {result.chunk.content}\n")