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
from rich.align import Align
from rich.markup import escape

sys.path.append(str(Path(__file__).parent))

import config_manager
import agents.prepare_agent as prepare_agent
from rag.core.base_embedder import BaseEmbedder
from rag.utils.switch_db import get_default_embedder, get_default_retriever, get_default_query_transformer
from rag.retrievers.qdrant_retriever import QdrantRetriever
from old.linear_pipeline import MainLinearPipeline
from pipeline.agent_workflow.concrete_workflow import ConcreteAgentWorkflow
from pipeline.agent_workflow.distillation_tool import LLMDistillationTool
from pipeline.agent_workflow.grep_tool import GrepTool
from pipeline.agent_workflow.script_finder_tool import PathScriptFinder
from pipeline.agent_workflow.rag_tool import SimpleRAGTool, AdvancedRAGTool
from pipeline.agent_workflow.file_tree_tool import FileTreeTool
from pipeline.agent_workflow.graph_tool import EnvisionGraphTool
from pipeline.agent_workflow.agentic_pipeline import AgenticPipeline
from pipeline.langgraph_base import BasePipeline, GraphState, BenchmarkState, APIError

class MainAgenticPipeline(AgenticPipeline):
    def __init__(self, console: Console, verbose=True):
        console.print("[bold green]🚀 Initializing...[/bold green]")
        
        self.config_manager = config_manager.get_config()
        self.agent = None
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)
        
        embedder = get_default_embedder()
        embedder.initialize()
        retriever = get_default_retriever()
        retriever.initialize(embedder.embedding_dimension)

        query_transformer = get_default_query_transformer()
        
        # Determine index type from flags and config
        index_type = self.config_manager.get("embedder.index_type", "full_chunk")
        if index_type == "full_chunk":
            index_path = Path("data/faiss_index")
        if index_type == "summary":
            index_path = Path("data/faiss_summary_index")
        if index_type == "raptor":
            index_path = Path("data/raptor_summary_index")
        
        retriever.load_index(str(index_path))
            
        self.rag = {'embedder': embedder, 'retriever': retriever, 'query_transformer': query_transformer,}
        if self.config_manager.get("main_pipeline.rag_tool.advanced", False):
            rag_tool = AdvancedRAGTool(retriever=retriever, embedder=embedder, query_transformer=query_transformer)
        else:
            rag_tool = SimpleRAGTool(retriever=retriever, embedder=embedder, query_transformer=query_transformer)
        grep_tool = GrepTool()
        graph_tool = EnvisionGraphTool()
        script_finder_tool = PathScriptFinder()
        tree_tool = FileTreeTool()
        distillation_tool = LLMDistillationTool(console=console)
        
        # Pre-compile the agent sub-graph to avoid recompiling on every node call
        agent_workflow = ConcreteAgentWorkflow(
            rag_tool, 
            grep_tool, 
            graph_tool,
            script_finder_tool, 
            distillation_tool,
            tree_tool
        )
        
        super().__init__(console, agent_workflow)

        self.console.print("[bold green]✅ Ready[/bold green]\n")
    
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
        
        elif self.config_manager.get_benchmark_type().startswith('llm_as_a_judge'):
            from pipeline.benchmarks.llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark, LLMAsAJudgeBenchmark2
            self.console.print("[dim]--- NODE: Judge LLM Grade Answer ---[/dim]")
            
            if self.config_manager.get_benchmark_type() == "llm_as_a_judge2":
                benchmark = LLMAsAJudgeBenchmark2()
            else:
                benchmark = LLMAsAJudgeBenchmark()
            benchmark.initialize()

            #delay to avoid too many requests
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
            qa_pair = {
                "question": state["question"],
                "llm_response": state["final_answer"],
                "reference": state["reference_answer"]
            }
            grade = benchmark.run([qa_pair])["results"][0]
            if state["verbose"]:
                self.console.print(f"[dim]→ LLM ({benchmark.agent.model_name}) Judge score with reference: {grade['score']}[/dim]")
            
            return {"grade": grade}

class DSLQuerySystem():
    def __init__(self, pipeline: BasePipeline, console: Console):
        self.config_manager = config_manager.get_config()
        self.pipeline = pipeline
        self.console = console

    def query(self, question, verbose=True):
        simple_qa_graph = self.pipeline.build_single_qa_graph()
        app = simple_qa_graph.compile()
        input_state = GraphState(question=question, verbose=verbose, reference_answer="", retry_count=0)
        try:
            final_state = app.invoke(input_state)
        finally:
            if isinstance(self.pipeline.rag["retriever"], QdrantRetriever):
                self.pipeline.rag["retriever"].close()  # Ensure Qdrant client is properly closed on exit
        return final_state.get("final_answer", "No answer generated")
            
    def interactive(self, verbose=False):
        simple_qa_graph = self.pipeline.build_single_qa_graph()
        app = simple_qa_graph.compile()
        
        self.console.print(Panel("Envision Copilot (Ctrl+C to exit)", title="Interactive", border_style="purple"))
        
        while True:
            try:
                if self.config_manager.get("main_pipeline.token_count", False):
                    self.config_manager.config["tokens_in"] = 0
                    self.config_manager.config["tokens_out"] = 0
                user_input = self.console.input("[bold purple]User:[/bold purple] ")
                if not user_input.strip(): continue
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if verbose:
                    self.console.print("[dim]Thinking...[/dim]")

                input_state = GraphState(question=user_input, verbose=verbose, reference_answer="", retry_count=0)
                final_state = app.invoke(input_state)
                raw = final_state.get('final_answer', 'No answer generated')
                
                self.console.print(Panel(Markdown(raw), title="Copilot", border_style="blue"))
                if self.config_manager.get("main_pipeline.token_count", False):
                    self.console.print(f"\n[bold]Tokens used: [/bold]{self.config_manager.get('tokens_in')} [green]tokens in[/green]"
                                       f", {self.config_manager.get('tokens_out')} [red]tokens out[/red]")
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        if isinstance(self.pipeline.rag["retriever"], QdrantRetriever):
            self.pipeline.rag["retriever"].close()  # Ensure Qdrant client is properly closed on exit
        self.console.print("\n[bold]👋 Goodbye![/bold]")
    
    def _display_and_save_results(
        self,
        grades: list,
        questions_json_path: str,
        interrupted: bool = False,
    ) -> None:
        """
        Display a results table and optionally save to JSON.
 
        Extracted so that it can be called both on normal completion and on
        partial results after an interruption.
 
        Args:
            grades:              List of grade dicts collected so far.
            questions_json_path: Original benchmark file path (used for the
                                 saved filename stem).
            interrupted:         Whether the run was cut short — affects the
                                 heading and the saved filename.
        """
        if not grades:
            self.console.print("[yellow]No results to display.[/yellow]")
            return
 
        heading = "# Benchmark Results (PARTIAL — interrupted)" if interrupted else "# Benchmark Results"
        self.console.print(Markdown(heading))
 
        table = Table(title="Benchmark Grades", show_lines=True)
        table.add_column("Question", style="cyan", no_wrap=False)
        table.add_column("Score", style="magenta")
 
        for r in grades:
            table.add_row(f"{r['id']}) " + r['question'], f"{r['score']:.4f}")
            self.console.print(f"\n[bold green]{r['id']}) Question: {escape(r['question'])} [/bold green]\n")
            self.console.print(f"[bold purple]  Référence: {escape(r['reference'])}[/bold purple]")
            self.console.print("\n[bold blue]  LLM: [/bold blue]")
            self.console.print(Markdown(f"{escape(r['llm_response'])}"))
            self.console.print(f"\n[bold red] Score : [/bold red]{escape(str(r['score']))}")
 
        self.console.print("\n")
        self.console.print(Align.center(table))
 
        avg = sum(r["score"] for r in grades) / len(grades)
        label = f"Moyenne partielle ({len(grades)} questions)" if interrupted else "Moyenne globale"
        self.console.print(f"\n[bold]{label} : {avg:.4f}[/bold]")
 
        if self.config_manager.get("main_pipeline.token_count", False):
            self.console.print(
                f"\n[bold]Tokens used: [/bold]{self.config_manager.get('tokens_in')} "
                f"[green]tokens in[/green], {self.config_manager.get('tokens_out')} [red]tokens out[/red]"
            )
 
        if self.config_manager.get("benchmark.save_data", False):
            data_dir = self.config_manager.get("paths.data_dir", "data")
            benchmark_name = Path(questions_json_path).stem
            suffix = "_partial" if interrupted else ""
            res_dir = (
                data_dir
                + f"/benchmark_results/{benchmark_name}"
                + suffix
                + f"_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            res_path = Path(res_dir)
            res_path.parent.mkdir(parents=True, exist_ok=True)
            res = {
                "Models": {
                    "Main agent": self.config_manager.get("agent.default_model"),
                    "Benchmark agent": self.config_manager.get("benchmark.benchmark_model")
                },
                "Results": grades
            }
            if self.config_manager.get("main_pipeline.token_count", False):
                res["Tokens used"] = {
                    "In": self.config_manager.get('tokens_in'),
                    "Out": self.config_manager.get('tokens_out'),
                }
            with open(res_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            self.console.print(f"\n[dim]Results saved to {res_path}[/dim]")
    
    def benchmark(self, questions_json_path: str, verbose=False,
                  start_from: int = 1):
        # Build the sub-graph for single Q/A processing
        sub_rag_system = self.pipeline.build_single_qa_graph()
        
        # Build the full benchmark graph
        workflow = self.pipeline.build_full_benchmark_graph()

        # Compile the graph
        app = workflow.compile()

        # Charger les questions
        with open(questions_json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        qa_pairs = [
            (q["question"], "\n".join([str(a) for a in q["answers"]]))
            for q in questions['answered']
        ]
        
        total = len(qa_pairs)
        
        # Validate start_from
        if start_from < 1 or start_from > total:
            self.console.print(
                f"[bold red]--benchmarkstart {start_from} is out of range "
                f"(1–{total}). Running from question 1.[/bold red]"
            )
            start_from = 1
 
        if start_from > 1:
            self.console.print(
                f"[yellow]Skipping questions 1–{start_from - 1}. "
                f"Starting from question {start_from}/{total}.[/yellow]"
            )
            
        input_state = BenchmarkState(
            qa_pairs=qa_pairs[start_from-1:],
            verbose=verbose,
            sub_rag_system=sub_rag_system
        )

        try:
            final_state = app.invoke(input_state)
        except APIError as exc:
            final_state = exc.saved_state
        finally:
            if isinstance(self.pipeline.rag["retriever"], QdrantRetriever):
                self.pipeline.rag["retriever"].close()  # Ensure Qdrant client is properly closed on exit
        
        # Put correct id for questions
        grades = final_state["grades"]
        for (i, r) in enumerate(grades):
            r["id"] = i+start_from
            
        self._display_and_save_results(grades, questions_json_path, len(grades) != total-start_from+1)


def main():
    """Main entry point."""

    
    console = Console()
    
    parser = argparse.ArgumentParser(
        description="DSL Query System - AI-powered code analysis and information extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  Interactive Mode (default): Start an interactive session for multiple queries
  Query Mode: Process a single query and exit
  Status Mode: Check system status and configuration
  Benchmark Mode: Evaluate system against test questions

EXAMPLES:
  # Interactive mode
  python main.py                                       # Start interactive mode
  python main.py --interactive                        # Explicit interactive mode
  python main.py --interactive --verbose              # Interactive with detailed output
  
  # Query mode
  python main.py --query "[QUERY]"             # Process single query
  python main.py --query "data flow" --verbose        # Query with detailed reasoning
  python main.py --query "find instances" --quiet     # Query with only result output
  
  # Status mode
  python main.py --status                             # Check system status and config
  
  # Agent selection
  python main.py --agent mistral --query "code analysis"      # Use specific agent
  python main.py --agent gpt --interactive            # Start interactive with GPT
  python main.py --agent gemini --query "find pattern"        # Query with Gemini
  python main.py --agent qwen --query "summarize"          # Query with local Qwen
  
  # Index type
  python main.py --indextype full_chunk --query "[QUERY]"  # Use full chunk index
  python main.py --indextype summary --interactive             # Use summary index
  
  # RAG fusion
  python main.py --fusion --query "[COMPLEX QUERY]"  # Enable RAG fusion for query
  python main.py --fusion --interactive               # Enable RAG fusion mode
  
  # Benchmark mode
  python main.py --benchmarkpath questions.json       # Run benchmark evaluation
  python main.py --benchmarkpath test.json --verbose  # Benchmark with detailed output
  python main.py --benchmarktype llm_as_a_judge --benchmarkagent gpt --benchmarkpath data.json   # Use LLM judge
  python main.py --benchmarktype cosine_similarity --benchmarkpath data.json # Use cosine similarity
  python main.py --benchmarkpath questions.json --benchmarkstart 5  # Resume from Q5
  
  # Linear mode
  python main.py --linear             # Untoggle agentic pipeline
  
  # Token count
  python main.py --token_count         # Get number of tokens used in and out
  
  # Save benchmark results
  python main.py -bp "a/path/to/benchmark" --save_data     # Saves the benchmark result in a json file in data folder

  # Combined options
  python main.py --agent mistral --fusion --query "find configs"            # Mistral + fusion
  python main.py --agentic --agent gpt --query "analyze code"               # Agentic with GPT
  python main.py --indextype summary --verbose --query "search knowledge"   # Summary index + verbose
  python main.py --benchmarkpath questions.json --benchmarkagent gemini --verbose  # Benchmark with details
  
  # Help
  python main.py --help                               # Show full help message
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

    mode_group.add_argument(
        "--indextype", "-in",
        choices=["full_chunk", "summary"],
        help="Override the type of index built from config"
        )
    
    # Linear Toggle
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Toggle the linear mode (disable agentic pipeline)"
    )
    
    # Agent selection
    parser.add_argument(
        "--agent", "-a",
        choices=["mistral", "groq", "qwen", "qwen-ssh"],
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
        "--querytransform", "-qt",
        choices=["fusion", "hyde"],
        help="Override query transform mode from config"
    )
    
    parser.add_argument(
        "--benchmarkpath", "-bp",
        metavar="PATH",
        help="Run benchmark with a JSON file containing questions and expected answers"
    )

    parser.add_argument(
        "--benchmarktype", "-bt",
        choices=["llm_as_a_judge", "llm_as_a_judge2", "cosine_similarity"],
        help="Override benchmark type from config"
    )

    parser.add_argument(
        "--benchmarkagent", "-ba",
        choices=["mistral", "groq", "qwen", "qwen-ssh"],
        help="Override benchmark agent from config"
    )
    
    parser.add_argument(
        "--benchmarkstart", "-bs",
        metavar="N",
        type=int,
        default=1,
        help=(
            "1-based index of the question to start the benchmark from. "
            "Questions before this index are skipped. "
            "Useful to resume a benchmark that was interrupted."
        )
    )
    
    parser.add_argument(
        "--token_count", "-tc",
        action="store_true",
        help="Get the total tokens used for LLM calls"
    )
    
    parser.add_argument(
        "--save_data", "-sd",
        action="store_true",
        help="If benchmark is used, the results are saved in a json summary"
    )

    args = parser.parse_args()
    
    # Handle conflicting options
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together")
    
    try:
        #Override benchmark agent if specified
        if args.indextype:
            config_manager.get_config().config['embedder']['index_type'] = args.indextype

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
                
            # Determine index type from flags and config
                index_type = config_manager.get_config().get("embedder.index_type")
                if index_type == "full_chunk":
                    index_path = Path("data/faiss_index")
                if index_type == "summaries": 
                    index_path = Path("data/faiss_summary_index")

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
            pipeline_logic = config_manager.get_config().config['main_pipeline']['agent_logic']
            pipeline_logic['distillation_llm'] = args.agent
            pipeline_logic['main_llm'] = args.agent
            pipeline_logic['planner_llm'] = args.agent
            pipeline_logic['cleaning_llm'] = args.agent

        if config_manager.get_config().get_default_agent() in ["qwen", "qwen-ssh"]:
            # Disable rate limiting for local LLM
            config_manager.get_config().config['agent']['rate_limit_delay'] = 0

        if args.linear != None:
            config_manager.get_config().config['main_pipeline']['agentic'] = not args.linear
        
        #Override benchmark type if specified
        if args.benchmarktype:
            config_manager.get_config().config['benchmark']['benchmark_type'] = args.benchmarktype

        #Override benchmark agent if specified
        if args.benchmarkagent:
            config_manager.get_config().config['benchmark']['benchmark_model'] = args.benchmarkagent
        
        if args.save_data:
            config_manager.get_config().config['benchmark']['save_data'] = args.save_data

        #Override query transform mode if specified
        if args.querytransform:
            config_manager.get_config().config['query_transformer']['query_transformer_type'] = args.querytransform
        
        if args.token_count:
            config_manager.get_config().config['main_pipeline']['token_count'] = args.token_count
        
        if config_manager.get_config().get("main_pipeline.token_count", False):
            config_manager.get_config().config['tokens_in'] = 0
            config_manager.get_config().config['tokens_out'] = 0
            
        # Determine verbosity level
        if args.verbose:
            verbose = True
        elif args.quiet:
            verbose = False
        else:
            # Default: silent for interactive mode, visible for query mode
            verbose = bool(args.query)
        
        # Create and initialize system
        if config_manager.get_config().get('main_pipeline.agentic', False):
            pipeline = MainAgenticPipeline(verbose=verbose, console=console)
        else:
            pipeline = MainLinearPipeline(verbose=verbose, console=console)
        system = DSLQuerySystem(pipeline, console)

        # Benchmark mode
        if args.benchmarkpath:
            system.benchmark(args.benchmarkpath, verbose=verbose,
                             start_from=args.benchmarkstart)
       
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
                if config_manager.get_config().get("main_pipeline.token_count", False):
                    console.print(f"\n[bold]Tokens used: [/bold]{config_manager.get_config().get('tokens_in')} "
                                  f"[green]tokens in[/green], {config_manager.get_config().get('tokens_out')} "
                                  "[red]tokens out[/red]")
        else:
            # Interactive mode (default) - always clean interface
            system.interactive(verbose=args.verbose)
            
    except KeyboardInterrupt:
        console.print("\n\n[bold]👋 Goodbye![/bold]")


if __name__ == "__main__":
    main()
