#!/usr/bin/env python3
"""DSL Query System"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from rag.core.session import QuerySession
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever
from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark 

# Dynamic agent imports - only import when needed


class DSLQuerySystem:
    def __init__(self):
        self.config = ConfigManager()
        self.router = None
        self.grep = None
        self.rag = {}
        self.agent = None
        
    def initialize(self, verbose=True):
        if verbose:
            print("🚀 Initializing...")
            
        agent_type = self.config.get_default_agent()
        if agent_type == 'mistral':
            from agents.mistral_agent import MistralAgent
            self.agent = MistralAgent()
        elif agent_type == 'gemini':
            from agents.gemini_agent import GeminiAgent
            self.agent = GeminiAgent()
        else:
            from agents.gpt_agent import GPTAgent
            self.agent = GPTAgent()
            
        self.agent.initialize()
        self.router = Router(self.agent)
        
        dirs = self.config.get('paths.input_dirs', ["env_scripts"])
        self.grep = GrepSearcher(dirs)
        
        parser = EnvisionParser(self.config.get_parser_config())
        chunker = SemanticChunker(self.config.get_chunker_config())
        embedder = SentenceTransformerEmbedder(self.config.get_embedder_config())
        embedder.initialize()
        retriever = FAISSRetriever(self.config.get_retriever_config())
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
            
    def query(self, question, verbose=False):
        c = self.router.classify(question)
        if verbose:
            print(f"🎯 {c.qtype.value} ({c.confidence:.0%})")
            
        if c.qtype == QueryType.GREP:
            r = self.grep.search(c.pattern or "")
            return self.grep.format_answer(r, question)
        else:
            emb = self.rag['embedder'].embed_text(question)
            results = self.rag['retriever'].search(emb, top_k=5)
            ctx = "\n\n".join([f"[{r.chunk.metadata.get('file_path', 'unknown')}]\n{r.chunk.content}" for r in results])
            return self.agent.generate_response(question, ctx)
            
    def interactive(self):
        print("\n💬 Interactive (exit to quit)")
        print("=" * 60)
        while True:
            try:
                q = input("\n❓ ").strip()
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                if q:
                    print(f"\n💡 {self.query(q, verbose=True)}")
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
        choices=["gemini", "gpt", "mistral"],
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
    "--benchmark",
    metavar="PATH",
    help="Run benchmark with a JSON file containing questions and expected answers"
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
        
        # Create and initialize system for query modes
        system = DSLQuerySystem()
        
        # Override agent if specified
        if args.agent:
            system.config_manager.config['agent'] = {'default_model': args.agent}
            
        # Determine verbosity level
        if args.verbose:
            verbose = True
        elif args.quiet:
            verbose = False
        else:
            # Default: silent for interactive mode, visible for query mode
            verbose = bool(args.query)
            
        system.initialize(verbose=verbose)
        # Benchmark mode
        if args.benchmark:
            import json
            from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark

            # Charger les questions
            with open(args.benchmark, "r", encoding="utf-8") as f:
                questions = json.load(f)

            embedder = system.embedder
            benchmark = CosineSimBenchmark(embedder)

            data_for_benchmark = []
            for q in questions:
                question = q["question"]
                reference = q["answer"]

                print(f"\n🔍 Question: {question}")
                llm_response = system.query(question, transparent=False)
                print(f"🤖 Réponse LLM: {llm_response}")
                print(f"🎯 Référence: {reference}")

                data_for_benchmark.append({
                    "question": question,
                    "llm_response": llm_response,
                    "reference": reference
                })

            report = benchmark.run(data_for_benchmark)

            print("\n📊 Résultats du benchmark Cosine Similarity")
            print("=" * 60)
            for r in report["results"]:
                print(f"Q: {r['question']}")
                print(f"→ Similarité: {r['similarity']:.4f}")
                print("-" * 40)

            print(f"\nMoyenne globale : {report['mean_score']:.4f}")
            return
       
        # Determine mode and execute
        if args.query:
            # Single query mode
            if args.quiet:
                # Just the response, no transparency
                response = system.query(args.query, transparent=False)
                print(response)
            else:
                # Full transparency if verbose, minimal if normal
                transparent = args.verbose
                response = system.query(args.query, transparent=transparent)
                if not transparent:
                    print(response)
        else:
            # Interactive mode (default) - always clean interface
            system.interactive_mode(verbose=args.verbose)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
