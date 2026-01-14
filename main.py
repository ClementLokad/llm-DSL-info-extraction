#!/usr/bin/env python3
"""DSL Query System"""
import sys
import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()

from transformers import pipeline

sys.path.append(str(Path(__file__).parent))

import config_manager
import agents.prepare_agent as prepare_agent
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.core.base_embedder import BaseEmbedder
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever
from rag.retrievers.grep_retriever import GrepRetriever
from rag.router import Router, QueryType
from rag.core.base_retriever import RetrievalResult
from pipeline.agent_workflow.concrete_workflow import ConcreteAgentWorkflow
from pipeline.agent_workflow.distillation_tool import LLMDistillationTool
from pipeline.agent_workflow.grep_tool import GrepTool
from pipeline.agent_workflow.script_finder_tool import PathScriptFinder
from pipeline.agent_workflow.rag_tool import SimpleRAGTool
from pipeline.agent_workflow.agentic_pipeline import AgenticPipeline
from langgraph_base import BasePipeline, GraphState, BenchmarkState

# Dynamic agent imports - only import when needed

def merge_rag_results(results):
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

# Main grading function used in both pipelines
def main_grade_answer(state: GraphState, embedder: BaseEmbedder):
    cm = config_manager.get_config()
    rate_limit_delay = cm.get('agent.rate_limit_delay', 0)
    final_answer = state["final_answer"]
    reference_answer = state["reference_answer"]
    
    if cm.get_benchmark_type() == 'cosine_similarity':
        from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark 
        if state["verbose"]:
            console.print("[dim]--- NODE: Cosine Similarity Grade Answer ---[/dim]")
        
        benchmark = CosineSimBenchmark(embedder)
        
        score = benchmark.compute_similarity(final_answer, reference_answer)
        if state["verbose"]:
            console.print(f"[dim]→ Similarity score with reference: {score:.4f}[/dim]")
        
        grade = {"score": score,
                "question": state["question"],
                "llm_response": state["final_answer"],
                "reference": state["reference_answer"]}
        
        return {"grade": grade}
    
    elif cm.get_benchmark_type() == 'llm_as_a_judge':
        from pipeline.benchmarks.llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark
        if state["verbose"]:
            console.print("[dim]--- NODE: Judge LLM Grade Answer ---[/dim]")
        
        benchmark = LLMAsAJudgeBenchmark()
        benchmark.initialize()

        #delay to avoid too many requests
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        
        score = benchmark.judge(state["question"], final_answer, reference_answer)
        
        if state["verbose"]:
            console.print(f"[dim]→ LLM Judge score with reference: {score}[/dim]")
        
        grade = {"score": score,
                "question": state["question"],
                "llm_response": state["final_answer"],
                "reference": state["reference_answer"]}
        
        return {"grade": grade}

class MainLinearPipeline(BasePipeline):
    def __init__(self, verbose=True):
        if verbose:
            console.print("[bold green]🚀 Initializing...[/bold green]")
        self.config_manager = config_manager.get_config()
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)

        self.agent = prepare_agent.prepare_default_agent()
        self.router = Router(self.agent)
        
        dirs = self.config_manager.get('paths.input_dirs', ["env_scripts"])
        self.grep = GrepRetriever(dirs)
        
        parser = EnvisionParser(self.config_manager.get_parser_config())
        chunker = SemanticChunker(self.config_manager.get_chunker_config())
        embedder = SentenceTransformerEmbedder(self.config_manager.get_embedder_config())
        embedder.initialize()
        retriever = FAISSRetriever(self.config_manager.get_retriever_config())
        retriever.initialize(embedder.embedding_dimension)
        
        index_path = Path("data/faiss_index")
        metadata_file = index_path / "metadata.pkl"
        
        if not metadata_file.exists():
            if verbose:
                console.print("[dim]Building index...[/dim]")
            blocks = []
            for d in dirs:
                p = Path(d)
                if p.exists():
                    for f in p.glob("*.nvn"):
                        blocks.extend(parser.parse_file(str(f)))
            chunks = chunker.chunk_blocks(blocks)
            embs = embedder.embed_chunks(chunks)
            retriever.add_chunks(chunks, embs)
            index_path.mkdir(parents=True, exist_ok=True)
            retriever.save_index(str(index_path))
        else:
            retriever.load_index(str(index_path))
            
        self.rag = {'embedder': embedder, 'retriever': retriever}

        if verbose:
            console.print("[bold green]✅ Ready[/bold green]\n")

    def retrieve_documents(self, state):
        if state["verbose"]:
            console.print("[dim]--- NODE: Retrieve Documents ---[/dim]")
        question = state["question"]
        retrieved_context = []
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        c = self.router.classify(question)
        if state["verbose"]:
            console.print(f"[dim]🎯 Router decision: {c.qtype.value} ({c.confidence:.0%} confidence)[/dim]")
        
        top_k = self.config_manager.get('rag.top_k_chunks', 5)
        
        if c.qtype == QueryType.GREP:
            retrieved_context = self.grep.search(c.pattern or "")
        elif self.config_manager.get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + question)
            if state["verbose"]:
                console.print(f"[dim]Raw answer from LLM for decomposition of the query : {raw_questions}[/dim]")
            questions = raw_questions.split("$")
            for sub_question in questions:
                emb = self.rag['embedder'].embed_text(sub_question)
                retrieved_context.extend(self.rag['retriever'].search(emb, top_k=top_k))
            retrieved_context = merge_rag_results(retrieved_context)[:top_k]
        else:
            emb = self.rag['embedder'].embed_text(question)
            retrieved_context = self.rag['retriever'].search(emb, top_k=top_k)
        
        if state["verbose"]:
            console.print(f"[dim]🔍 → Retrieved {len(retrieved_context)} documents :[/dim]")
            # console.print(retrieved_context)

        return {"retrieved_context": retrieved_context}
    
    def engineer_prompt(self, state):
        if state["verbose"]:
            console.print("[dim]--- NODE: Engineer Prompt ---[/dim]")
        
        question = state["question"]
        context = state["retrieved_context"]
        
        ctx: str
        if len(context) ==0:
            ctx = "No relevant context found."
        else:
            ctx = "\n\n----------------------\n\n".join([r.to_str_for_generation() for r in context])
        
        prompt = f"Given this context:\n{ctx}\n________________________\n\nAnswer the following question:\n{question}"
        
        if state["verbose"]:
            console.print(f"[dim]→ Generated prompt size: {len(prompt)} chars[/dim]")
        
        return {"prompt": prompt}
    
    def generate_answer(self, state):
        if state["verbose"]:
            console.print("[dim]--- NODE: Generate Answer (Main LLM) ---[/dim]")
        prompt = state["prompt"]
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        generation = self.agent.generate_response(prompt)
        
        if state["verbose"]:
            console.print(f"[dim]💬 → LLM RAW Generation complete[/dim]")
        
        return {"generation": generation}
    
    def grade_answer(self, state):
        return main_grade_answer(state, self.rag['embedder'])

class MainAgenticPipeline(AgenticPipeline):
    def __init__(self, verbose=True):
        if verbose:
            console.print("[bold green]🚀 Initializing...[/bold green]")
        
        self.config_manager = config_manager.get_config()
        self.agent = None
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)
        
        dirs = self.config_manager.get('paths.input_dirs', ["env_scripts"])
        
        parser = EnvisionParser(self.config_manager.get_parser_config())
        chunker = SemanticChunker(self.config_manager.get_chunker_config())
        embedder = SentenceTransformerEmbedder(self.config_manager.get_embedder_config())
        embedder.initialize()
        retriever = FAISSRetriever(self.config_manager.get_retriever_config())
        retriever.initialize(embedder.embedding_dimension)
        
        index_path = Path("data/faiss_index")
        metadata_file = index_path / "metadata.pkl"
        
        if not metadata_file.exists():
            if verbose:
                console.print("[dim]Building index...[/dim]")
            blocks = []
            for d in dirs:
                p = Path(d)
                if p.exists():
                    for f in p.glob("*.nvn"):
                        blocks.extend(parser.parse_file(str(f)))
            chunks = chunker.chunk_blocks(blocks)
            embs = embedder.embed_chunks(chunks)
            retriever.add_chunks(chunks, embs)
            index_path.mkdir(parents=True, exist_ok=True)
            retriever.save_index(str(index_path))
        else:
            retriever.load_index(str(index_path))
            
        self.rag = {'embedder': embedder, 'retriever': retriever}
            
        rag_tool = SimpleRAGTool(retriever=retriever, embedder=embedder)
        grep_tool = GrepTool()
        script_finder_tool = PathScriptFinder()
        distillation_tool = LLMDistillationTool()
        
        # Pre-compile the agent sub-graph to avoid recompiling on every node call
        agent_workflow = ConcreteAgentWorkflow(
            rag_tool, 
            grep_tool, 
            script_finder_tool, 
            distillation_tool
        )
        
        super().__init__(agent_workflow)

        if verbose:
            console.print("[bold green]✅ Ready[/bold green]\n")
    
    def grade_answer(self, state):
        return main_grade_answer(state, self.rag['embedder'])

class DSLQuerySystem():
    def __init__(self, pipeline: BasePipeline):
        self.config_manager = config_manager.get_config()
        self.pipeline = pipeline

    def query(self, question, verbose=True):
        simple_qa_graph = self.pipeline.build_single_qa_graph()
        app = simple_qa_graph.compile()
        input_state = GraphState(question=question, verbose=verbose, reference_answer="", retry_count=0)
        final_state = app.invoke(input_state)
        return final_state.get("final_answer", "No answer generated")
            
    def interactive(self, verbose=False):
        simple_qa_graph = self.pipeline.build_single_qa_graph()
        app = simple_qa_graph.compile()
        
        console.print(Panel("Envision Copilot (Ctrl+C to exit)", title="Interactive", border_style="purple"))
        
        while True:
            try:
                user_input = console.input("[bold purple]User:[/bold purple] ")
                if not user_input.strip(): continue
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if verbose:
                    console.print("[dim]Thinking...[/dim]")

                input_state = GraphState(question=user_input, verbose=verbose, reference_answer="", retry_count=0)
                final_state = app.invoke(input_state)
                raw = final_state.get('final_answer', 'No answer generated')
                answer = extract_answer(raw)
                
                console.print(Panel(Markdown(answer), title="Copilot", border_style="blue"))
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("\n[bold]👋 Goodbye![/bold]")
    
    def benchmark(self, questions_json_path: str, verbose=False):
        # Build the sub-graph for single Q/A processing
        sub_rag_system = self.pipeline.build_single_qa_graph()
        
        # Build the full benchmark graph
        workflow = self.pipeline.build_full_benchmark_graph()

        # Compile the graph
        app = workflow.compile()

        # Charger les questions
        with open(questions_json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        input_state = BenchmarkState(
            qa_pairs=[(q["question"], "\n".join([str(a) for a in q["answers"]])) for q in questions['answered']],
            verbose=verbose,
            sub_rag_system=sub_rag_system
        )

        final_state = app.invoke(input_state)

        console.print(Panel("Benchmark Results", title="Results", border_style="green"))
        
        table = Table(title="Benchmark Grades")
        table.add_column("Question", style="cyan", no_wrap=False)
        table.add_column("Score", style="magenta")
        
        for r in final_state["grades"]:
            table.add_row(r['question'], f"{r['score']:.4f}")
            if verbose:
                console.print(f"[dim]  Ref: {r['reference'][:100]}...[/dim]")
                console.print(f"[dim]  LLM: {r['llm_response'][:100]}...[/dim]")

        console.print(table)
        console.print(f"\n[bold]Moyenne globale : {final_state['benchmark_results']['average_score']:.4f}[/bold]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DSL Query System - AI-powered code analysis and information extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  Interactive Mode (default): Start an interactive session for multiple queries
  Query Mode: Process a single query and exit
  Status Mode: Check system status and configuration

EXAMPLES:
  python main.py                          # Start interactive mode
  python main.py --interactive            # Explicit interactive mode
  python main.py --query "business logic" # Process single query
  python main.py --status                 # Check system status
  python main.py --agent mistral --query "data flow" # Use specific agent
  python main.py --help                   # Show this help
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode (default)"
    )
    
    mode_group.add_argument(
        "--query",
        metavar="QUERY",
        help="Process a single query and exit"
    )
    
    mode_group.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show system status and configuration"
    )
    
    # Agentic Toggle
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Toggle the agentic mode"
    )
    
    # Agent selection
    parser.add_argument(
        "--agent", "-a",
        choices=["gemini", "gpt", "mistral", "llama3", "groq", "qwen"],
        help="Override default agent from config"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress initialization messages"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed processing steps"
    )

    parser.add_argument(
        "--fusion", "-f",
        action="store_true",
        help="Enable RAG fusion"
    )
    
    parser.add_argument(
        "--benchmarkpath", "-bp",
        metavar="PATH",
        help="Run benchmark with a JSON file containing questions and expected answers"
    )

    parser.add_argument(
        "--benchmarktype", "-bt",
        choices=["llm_as_a_judge", "cosine_similarity"],
        help="Override benchmark type from config"
    )

    parser.add_argument(
        "--benchmarkagent", "-ba",
        choices=["gemini", "gpt", "mistral", "llama3", "groq", "qwen"],
        help="Override benchmark agent from config"
    )

    args = parser.parse_args()
    
    # Handle conflicting options
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together")
    
    try:
        # Status mode - lightweight check without full initialization
        if args.status:
            console.print("[bold]🔍 SYSTEM STATUS CHECK[/bold]")
            
            # Check configuration
            try:
                from config_manager import ConfigManager
                config_mgr = ConfigManager()
                default_agent = config_mgr.get_default_agent()
                console.print(f"✅ Configuration loaded")
                console.print(f"   Default agent: [cyan]{default_agent}[/cyan]")

                # Check if API keys are configured
                try:
                    if default_agent == 'mistral':
                        api_key = config_mgr.get_api_key('MISTRAL_API_KEY')
                    elif default_agent == 'gpt':
                        api_key = config_mgr.get_api_key('OPENAI_API_KEY')
                    elif default_agent == 'gemini':
                        api_key = config_mgr.get_api_key('GEMINI_API_KEY')
                    
                    if api_key:
                        console.print(f"   ✅ API key configured for {default_agent}")
                    else:
                        console.print(f"   [yellow]⚠️ API key missing for {default_agent}[/yellow]")
                except:
                    console.print(f"   [yellow]⚠️ API key check failed for {default_agent}[/yellow]")
                    
            except Exception as e:
                console.print(f"[bold red]❌ Configuration error:[/bold red] {e}")
                return
                
            # Check index status
            try:
                import os
                index_path = "data/faiss_index"
                if os.path.exists(index_path):
                    files = os.listdir(index_path)
                    console.print(f"✅ Index found: {len(files)} files")
                else:
                    console.print("[yellow]⚠️ No index found - run build_index.py first[/yellow]")
            except Exception as e:
                console.print(f"[bold red]❌ Index check failed:[/bold red] {e}")
                
            console.print("\n[dim]💡 Use --help for available commands[/dim]")
            return
        
        # Override agent if specified
        if args.agent:
            config_manager.get_config().config['agent']['default_model'] = args.agent
        
        if config_manager.get_config().get_default_agent() == 'llama3':
            # Disable rate limiting for local Llama 3
            config_manager.get_config().config['agent']['rate_limit_delay'] = 0
        
        if args.fusion:
           config_manager.get_config().config['rag']['fusion'] = True

        #Override benchmark type if specified
        if args.benchmarktype:
            config_manager.get_config().config['benchmark']['benchmark_type'] = args.benchmarktype

        #Override benchmark agent if specified
        if args.benchmarkagent:
            config_manager.get_config().config['benchmark']['benchmark_agent'] = args.benchmarkagent
            
        # Determine verbosity level
        if args.verbose:
            verbose = True
        elif args.quiet:
            verbose = False
        else:
            # Default: silent for interactive mode, visible for query mode
            verbose = bool(args.query)
        
        # Create and initialize system
        if args.agentic:
            pipeline = MainAgenticPipeline(verbose=verbose)
        else:
            pipeline = MainLinearPipeline(verbose=verbose)
        system = DSLQuerySystem(pipeline)

        # Benchmark mode
        if args.benchmarkpath:
            system.benchmark(args.benchmarkpath, verbose=verbose)
       
        # Determine mode and execute
        elif args.query:
            
            # Single query mode
            if args.quiet:
                # Just the response, no transparency
                response = system.query(args.query, verbose=False)
                print(response) # Keep raw print for piping
            else:
                # Full transparency if verbose
                if verbose:
                    console.print("[dim]Reasoning...[/dim]")
                response = system.query(args.query, verbose=args.verbose)
                
                console.print(Panel(Markdown(response), title="Copilot Result", border_style="blue"))
        else:
            # Interactive mode (default) - always clean interface
            system.interactive(verbose=args.verbose)
            
    except KeyboardInterrupt:
        console.print("\n\n[bold]👋 Goodbye![/bold]")


if __name__ == "__main__":
    main()
