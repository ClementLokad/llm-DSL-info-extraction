from typing import List
from pipeline.agent_workflow.workflow_base import BaseRAGTool
from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_embedder import BaseEmbedder
from rag.core.base_query_transformer import BaseQueryTransformer
from config_manager import get_config
from agents.prepare_agent import prepare_default_agent
from rag.utils.script_scanner import replace_constants_in_script
import time

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
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), verbose = False) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []

        # Query transform mode: transform the query before retrieval
        if self.query_transformer:
            transformed_question_list = self.query_transformer.transform(query)

            if verbose:
                self.console.print(f"[dim]Raw answer from LLM after query transformation : {', '.join(transformed_question_list)}[/dim]")
            
            for sub_question in transformed_question_list:
                emb = self.embedder.embed_text(sub_question)
                results.extend(self.retriever.search(emb, top_k=top_k))
            results = self.merge_rag_results(results)[:top_k]
        
        # Normal mode: Retrieve via the embedding of the query itself
        else:
            
            emb = self.embedder.embed_text(query)
            results = self.retriever.search(emb, top_k=top_k)
        
        # Replace constants in result
        for result in results:
           cleaned_content = replace_constants_in_script(result.chunk.content, script_path=result.chunk.metadata["file_path"])
           result.chunk.content = cleaned_content

        return results