import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from typing import List

import config_manager
import agents.prepare_agent as prepare_agent
from rag.core.base_embedder import BaseEmbedder
from rag.utils.switch_db import get_default_embedder, get_default_retriever
from rag.retrievers.grep_retriever import GrepRetriever
from rag.router import Router, QueryType
from rag.core.base_retriever import RetrievalResult
from pipeline.langgraph_base import BasePipeline, GraphState

def merge_rag_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
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


class MainLinearPipeline(BasePipeline):
    def __init__(self, console: Console, verbose=True):
        super().__init__(console)
        self.console.print("[bold green]🚀 Initializing...[/bold green]")
        self.config_manager = config_manager.get_config()
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)

        self.agent = prepare_agent.prepare_default_agent()
        self.router = Router(self.agent)
        
        dirs = self.config_manager.get('paths.input_dirs', ["env_scripts"])
        self.grep = GrepRetriever(dirs)

        embedder = get_default_embedder()
        embedder.initialize()
        retriever = get_default_retriever()
        retriever.initialize(embedder.embedding_dimension)
        
        # Determine index type from flags and config
        index_type = self.config_manager.get("embedder.index_type", "full_chunk")
        if index_type == "full_chunk":
            index_path = Path("data/faiss_index")
        if index_type == "summary":
            index_path = Path("data/faiss_summary_index")
        
        retriever.load_index(str(index_path))
            
        self.rag = {'embedder': embedder, 'retriever': retriever}

        self.console.print("[bold green]✅ Ready[/bold green]\n")

    def retrieve_documents(self, state):
        self.console.print("[dim]--- NODE: Retrieve Documents ---[/dim]")
        question = state["question"]
        retrieved_context = []
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        c = self.router.classify(question)
        self.console.print(f"[dim]🎯 Router decision: {c.qtype.value} ({c.confidence:.0%} confidence)[/dim]")
        
        top_k = self.config_manager.get('rag.top_k_chunks', 5)
        
        if c.qtype == QueryType.GREP:
            retrieved_context = self.grep.search(c.pattern or "")
        elif self.config_manager.get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + question)
            if state["verbose"]:
                self.console.print(f"[dim]Raw answer from LLM for decomposition of the query : {raw_questions}[/dim]")
            questions = raw_questions.split("$")
            for sub_question in questions:
                emb = self.rag['embedder'].embed_text(sub_question)
                retrieved_context.extend(self.rag['retriever'].search(emb, top_k=top_k))
            retrieved_context = merge_rag_results(retrieved_context)[:top_k]
        else:
            emb = self.rag['embedder'].embed_text(question)
            retrieved_context = self.rag['retriever'].search(emb, top_k=top_k)
        
        self.console.print(f"[dim]🔍 → Retrieved {len(retrieved_context)} documents :[/dim]")
        if state["verbose"]:
            self.console.print(escape(retrieved_context))

        return {"retrieved_context": retrieved_context}
    
    def engineer_prompt(self, state):
        self.console.print("[dim]--- NODE: Engineer Prompt ---[/dim]")
        
        question = state["question"]
        context = state["retrieved_context"]
        
        ctx: str
        if len(context) == 0:
            ctx = "No relevant context found."
            prompt = f"Given this context:\n{ctx}\n________________________\n\nAnswer the following question:\n{question}"
        
        else:
            # Check if likely a grep result to apply specific statistics behavior
            # Grep results usually have chunk_type='grep_match' or 'smart_reference'
            is_grep = len(context) > 0 and context[0].chunk.chunk_type in ['grep_match', 'smart_reference']

            if is_grep:
                # Stats calculation for precise counting tasks
                total_hits = len(context)
                unique_files_set = set(r.chunk.metadata.get('original_file_path', 'unknown') for r in context)
                unique_files = list(unique_files_set)
                unique_files.sort()
                unique_files_count = len(unique_files)
                
                # Group content by file to present a distinct list to the LLM
                grouped_content = {}
                for r in context:
                    fpath = r.chunk.metadata.get('original_file_path', 'unknown')
                    if fpath not in grouped_content:
                        grouped_content[fpath] = []
                    
                    # Annotate content with resolution info if available
                    content_str = r.chunk.content
                    if r.chunk.metadata.get('resolved_path'):
                         content_str += f"   (System resolved: {r.chunk.metadata['resolved_path']})"
                    
                    grouped_content[fpath].append(content_str)
                
                # Construct a clear, deduplicated context string
                context_parts = []
                for fpath in unique_files:
                    snippets = "\n".join([f"  - {s}" for s in grouped_content.get(fpath, [])])
                    context_parts.append(f"[File: {fpath}]\n{snippets}")
                
                context_str = "\n\n".join(context_parts)

                # Construct the context string with explicit instructions and stats
                stats_header = (
                    f"SEARCH REPORT:\n"
                    f"The system has performed a rigorous search resolving all variables (constants).\n"
                    f"The findings below are verified matches. Do not exclude any entry.\n"
                    f"- Total occurrences found: {total_hits}\n"
                    f"- Distinct scripts/files involved: {unique_files_count}\n"
                    f"\nDETAILED FINDINGS (Grouped by file):\n"
                    f"----------------------\n"
                )
                
                ctx = stats_header + context_str
                prompt = f"Given this context:\n{ctx}\n________________________\n\nAnswer the following question based mainly on the SEARCH REPORT statistics above:\n{question}"
            
            else:
                # Standard RAG context formatting
                context_str = "\n\n----------------------\n\n".join([r.to_str_for_generation() for r in context])
                ctx = context_str
                prompt = f"Given this context:\n{ctx}\n________________________\n\nAnswer the following question:\n{question}"
        
        self.console.print(f"[dim]→ Generated prompt size: {len(prompt)} chars[/dim]")
        
        return {"prompt": prompt}
    
    def generate_answer(self, state):
        self.console.print("[dim]--- NODE: Generate Answer (Main LLM) ---[/dim]")
        prompt = state["prompt"]
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        generation = self.agent.generate_response(prompt)
        
        self.console.print(f"[dim]💬 → LLM RAW Generation complete[/dim]")
        
        return {"generation": generation}
    
    def grade_answer(self, state: GraphState) -> GraphState:
        embedder = self.rag["embedder"]
        rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)
        final_answer = state["final_answer"]
        reference_answer = state["reference_answer"]
        
        if self.config_manager.get_benchmark_type() == 'cosine_similarity':
            from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark 
            self.console.print("[dim]--- NODE: Cosine Similarity Grade Answer ---[/dim]")
            
            benchmark = CosineSimBenchmark(embedder)
            
            score = benchmark.compute_similarity(final_answer, reference_answer)
            if state["verbose"]:
                self.console.print(f"[dim]→ Similarity score with reference: {score:.4f}[/dim]")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}
        
        elif self.config_manager.get_benchmark_type() == 'llm_as_a_judge':
            from pipeline.benchmarks.llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark
            self.console.print("[dim]--- NODE: Judge LLM Grade Answer ---[/dim]")
            
            benchmark = LLMAsAJudgeBenchmark()
            benchmark.initialize()

            #delay to avoid too many requests
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
            
            try:
                score = benchmark.judge(state["question"], final_answer, reference_answer)
            except Exception:
                self.console.print("[bold red]Error during LLM judging, defaulting score to[/bold red] 0")
                score = 0
            if state["verbose"]:
                self.console.print(f"[dim]→ LLM Judge score with reference: {score}[/dim]")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}