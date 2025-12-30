#!/usr/bin/env python3
"""DSL Query System"""
import sys
import argparse
import json
import time
from pathlib import Path

from transformers import pipeline

sys.path.append(str(Path(__file__).parent))

import config_manager
import agents.prepare_agent as prepare_agent
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever
from rag.retrievers.grep_retriever import GrepRetriever
from rag.router import Router, QueryType
from rag.core.base_retriever import RetrievalResult
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

class DSLQuerySystem(BasePipeline):
    def __init__(self):
        self.config_manager = config_manager.get_config()
        self.router = None
        self.grep = None
        self.rag = {}
        self.agent = None
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)
        self.benchmark_type= self.config_manager.get_benchmark_type()
        
    def initialize(self, verbose=True):
        if verbose:
            print("🚀 Initializing...")
            
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
                print("Building index...")
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
            print("✅ Ready\n")
    
    def retrieve_documents(self, state):
        print("--- NODE: Retrieve Documents ---")
        question = state["question"]
        retrieved_context = []
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        c = self.router.classify(question)
        print(f"🎯 Router decision: {c.qtype.value} ({c.confidence:.0%} confidence)")
        
        top_k = self.config_manager.get('rag.top_k_chunks', 5)
        
        if c.qtype == QueryType.GREP:
            retrieved_context = self.grep.search(c.pattern or "")
        elif self.config_manager.get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + question)
            if state["verbose"]:
                print(f"Raw answer from LLM for decomposition of the query : {raw_questions}")
            questions = raw_questions.split("$")
            for sub_question in questions:
                emb = self.rag['embedder'].embed_text(sub_question)
                retrieved_context.extend(self.rag['retriever'].search(emb, top_k=top_k))
            retrieved_context = merge_rag_results(retrieved_context)[:top_k]
        else:
            emb = self.rag['embedder'].embed_text(question)
            retrieved_context = self.rag['retriever'].search(emb, top_k=top_k)
        
        if state["verbose"]:
            print(f"🔍 → Retrieved {len(retrieved_context)} documents :")
            print(retrieved_context)

        return {"retrieved_context": retrieved_context}
    
    def engineer_prompt(self, state):
        print("--- NODE: Engineer Prompt ---")
        
        question = state["question"]
        context = state["retrieved_context"]
        
        ctx: str
        if len(context) ==0:
            ctx = "No relevant context found."
        else:
            ctx = "\n\n----------------------\n\n".join([r.to_str_for_generation() for r in context])
        
        prompt = f"Given this context:\n{ctx}\n________________________\n\nAnswer the following question:\n{question}"
        
        if state["verbose"]:
            print(f"→ Generated prompt:\n{prompt}\n")
        
        return {"prompt": prompt}
    
    def generate_answer(self, state):
        print("--- NODE: Generate Answer (Main LLM) ---")
        prompt = state["prompt"]
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        generation = self.agent.generate_response(prompt)
        
        if state["verbose"]:
            print(f"💬 → LLM RAW Generation:\n{generation}\n")
        
        return {"generation": generation}
    
    def grade_answer(self, state):
        final_answer = state["final_answer"]
        reference_answer = state["reference_answer"]
        if self.benchmark_type == 'cosine_similarity':
            from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark 
            print("--- NODE: Cosine Similarity Grade Answer ---")
            
            benchmark = CosineSimBenchmark(self.rag['embedder'])
            
            score = benchmark.compute_similarity(final_answer, reference_answer)
            if state["verbose"]:
                print(f"→ Similarity score with '{reference_answer}': {score:.4f}")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}
        
        elif self.benchmark_type == 'llm_as_a_judge':
            from pipeline.benchmarks.llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark
            print("--- NODE: Judge LLM Grade Answer ---")
            
            benchmark = LLMAsAJudgeBenchmark()
            benchmark.initialize()

            #delay to avoid too many requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            score = benchmark.judge(state["question"], final_answer, reference_answer)
            
            if state["verbose"]:
                print(f"→ LLM Judge score with '{reference_answer}': {score}")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}

    def query(self, question, verbose=True):
        simple_qa_graph = self.build_single_qa_graph()
        app = simple_qa_graph.compile()
        input_state = GraphState(question=question, verbose=verbose, reference_answer="")
        final_state = app.invoke(input_state)
        return final_state.get("final_answer", "No answer generated")
            
    def interactive(self, verbose=False):
        simple_qa_graph = self.build_single_qa_graph()
        app = simple_qa_graph.compile()
        print("\n💬 Interactive (exit to quit)")
        print("=" * 60)
        while True:
            try:
                q = input("\n❓ ").strip()
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                if q:
                    input_state = GraphState(question=q, verbose=verbose, reference_answer="")
                    final_state = app.invoke(input_state)
                    print(f"\n💡 {final_state.get('final_answer', 'No answer generated')}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ {e}")
        print("\n👋")


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
    
    # Agent selection
    parser.add_argument(
        "--agent", "-a",
        choices=["gemini", "gpt", "mistral", "llama3", "groq"],
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
        choices=["gemini", "gpt", "mistral", "llama3"],
        help="Override benchmark agent from config"
    )

    args = parser.parse_args()
    
    # Handle conflicting options
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together")
    
    try:
        # Status mode - lightweight check without full initialization
        if args.status:
            print("🔍 SYSTEM STATUS CHECK")
            print("=" * 50)
            
            # Check configuration
            try:
                from config_manager import ConfigManager
                config_mgr = ConfigManager()
                default_agent = config_mgr.get_default_agent()
                print(f"✅ Configuration loaded")
                print(f"   Default agent: {default_agent}")

                # Check if API keys are configured
                try:
                    if default_agent == 'mistral':
                        api_key = config_mgr.get_api_key('MISTRAL_API_KEY')
                    elif default_agent == 'gpt':
                        api_key = config_mgr.get_api_key('OPENAI_API_KEY')
                    elif default_agent == 'gemini':
                        api_key = config_mgr.get_api_key('GEMINI_API_KEY')
                    
                    if api_key:
                        print(f"   ✅ API key configured for {default_agent}")
                    else:
                        print(f"   ⚠️ API key missing for {default_agent}")
                except:
                    print(f"   ⚠️ API key check failed for {default_agent}")
                    
            except Exception as e:
                print(f"❌ Configuration error: {e}")
                return
                
            # Check index status
            try:
                import os
                index_path = "data/faiss_index"
                if os.path.exists(index_path):
                    files = os.listdir(index_path)
                    print(f"✅ Index found: {len(files)} files")
                else:
                    print("⚠️ No index found - run build_index.py first")
            except Exception as e:
                print(f"❌ Index check failed: {e}")
                
            print("\n💡 Use --help for available commands")
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
        
        # Create and initialize system for query modes
        system = DSLQuerySystem()   
        system.initialize(verbose=verbose)

        # Benchmark mode
        if args.benchmarkpath:
            # Build the sub-graph for single Q/A processing
            sub_rag_system = system.build_single_qa_graph()
            
            # Build the full benchmark graph
            workflow = system.build_full_benchmark_graph()

            # Compile the graph
            app = workflow.compile()

            # Charger les questions
            with open(args.benchmarkpath, "r", encoding="utf-8") as f:
                questions = json.load(f)
                
            input_state = BenchmarkState(
                qa_pairs=[(q["question"], q["answers"]) for q in questions['answered']],
                verbose=verbose,
                sub_rag_system=sub_rag_system
            )

            final_state = app.invoke(input_state)

            print("\n📊 Résultats du benchmark")
            print("=" * 60)
            for r in final_state["grades"]:
                print(f"Q: {r['question']}")
                if verbose:
                    print(f"  Référence : {r['reference']}")
                    print(f"  LLM Response: {r['llm_response']}")
                print(f"→ Score: {r['score']:.4f}")
                print("\n" + "-" * 40 + "\n")

            print(f"\nMoyenne globale : {final_state['benchmark_results']['average_score']:.4f}")
            return
       
        # Determine mode and execute
        if args.query:
            
            # Single query mode
            if args.quiet:
                # Just the response, no transparency
                response = system.query(args.query, verbose=False)
                print(response)
            else:
                # Full transparency if verbose
                response = system.query(args.query, verbose=args.verbose)
                print(response)
        else:
            # Interactive mode (default) - always clean interface
            system.interactive(verbose=args.verbose)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
