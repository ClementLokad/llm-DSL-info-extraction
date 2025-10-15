#!/usr/bin/env python3
"""
DSL Query System - Main Interface
A sophisticated system for querying Domain Specific Language (DSL) code using AI agents.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config_manager import ConfigManager
from pipeline.core.session import QuerySession
from pipeline.parsers.envision_parser import EnvisionParser
from pipeline.chunkers.semantic_chunker import SemanticChunker
from pipeline.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from pipeline.retrievers.faiss_retriever import FAISSRetriever

# Dynamic agent imports - only import when needed


class DSLQuerySystem:
    """Main DSL Query System with AI-powered code analysis."""
    
    def __init__(self):
        """Initialize the DSL Query System."""
        self.config_manager = ConfigManager()
        self.parser = None
        self.chunker = None  
        self.embedder = None
        self.retriever = None
        self.agent = None
        self.sessions = []
        
    def initialize(self, verbose: bool = True) -> None:
        """Initialize all system components."""
        if verbose:
            print("🚀 INITIALIZING DSL QUERY SYSTEM")
            print("=" * 60)
            
        try:
            # Initialize pipeline components
            if verbose:
                print("🔧 Initializing pipeline components...")
                
            self._initialize_pipeline(verbose)
            
            # Initialize AI agent
            if verbose:
                print("🤖 Initializing AI agent...")
                
            self._initialize_agent(verbose)
            
            # Load index
            if verbose:
                print("📚 Loading knowledge index...")
                
            self._load_index(verbose)
            
            if verbose:
                print("✅ System initialized and ready")
                print("=" * 60)
                
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            raise
            
    def _initialize_pipeline(self, verbose: bool = True) -> None:
        """Initialize parser, chunker, embedder, and retriever."""
        # Parser
        parser_config = self.config_manager.get_parser_config()
        self.parser = EnvisionParser(parser_config)
        if verbose:
            extensions = parser_config.get('supported_extensions', [])
            print(f"   ✅ Parser: {extensions}")
        
        # Chunker  
        chunker_config = self.config_manager.get_chunker_config()
        self.chunker = SemanticChunker(chunker_config)
        if verbose:
            max_tokens = chunker_config.get('max_chunk_tokens', 512)
            print(f"   ✅ Chunker: max_tokens={max_tokens}")
        
        # Embedder
        embedder_config = self.config_manager.get_embedder_config()
        self.embedder = SentenceTransformerEmbedder(embedder_config)
        self.embedder.initialize()  # Initialize the embedder
        if verbose:
            model = embedder_config.get('sentence_transformer', {}).get('model_name', 'unknown')
            print(f"   ✅ Embedder: model={model}")
        
        # Retriever
        retriever_config = self.config_manager.get_retriever_config()
        self.retriever = FAISSRetriever(retriever_config)
        # Initialize with embedder's dimension
        embedding_dim = self.embedder.embedding_dimension
        self.retriever.initialize(embedding_dim)
        if verbose:
            metric = retriever_config.get('faiss', {}).get('similarity_metric', 'N/A')
            print(f"   ✅ Retriever: metric={metric}")
            
    def _initialize_agent(self, verbose: bool = True) -> None:
        """Initialize the configured AI agent."""
        default_agent = self.config_manager.get_default_agent()
        
        try:
            if default_agent == 'gemini':
                from agents.gemini_agent import GeminiAgent
                self.agent = GeminiAgent()
            elif default_agent == 'gpt':
                from agents.gpt_agent import GPTAgent
                self.agent = GPTAgent()
            elif default_agent == 'mistral':
                from agents.mistral_agent import MistralAgent
                self.agent = MistralAgent()
            else:
                raise ValueError(f"Unknown agent: {default_agent}")
                
            self.agent.initialize()
            
            if verbose:
                print(f"   ✅ {self.agent.model_name} initialized")
                
        except Exception as e:
            if verbose:
                print(f"   ⚠️ {default_agent} agent failed: {e}")
            raise
            
    def _load_index(self, verbose: bool = True) -> None:
        """Load the FAISS index, building it if necessary."""
        index_path = Path("data/faiss_index")
        
        # Check if index exists and has required files
        required_files = ["metadata.pkl", "chunks.pkl"]
        index_exists = index_path.exists() and all((index_path / f).exists() for f in required_files)
        
        if not index_exists:
            if verbose:
                print("⚠️ Index not found, building it now...")
            self._build_index(verbose)
            
        try:
            index_info = self.retriever.load_index(str(index_path))
        except Exception as e:
            if verbose:
                print(f"⚠️ Index loading failed ({e}), rebuilding...")
            self._build_index(verbose)
            # After building, we don't need to reload since _build_index already sets up the retriever
            index_info = {'status': 'built', 'total_vectors': 'N/A', 'total_chunks': 'N/A', 'embedding_dimension': 'N/A'}
            
        if index_info is None:
            # After building, create a dummy info dict
            index_info = {'status': 'built', 'total_vectors': 'N/A', 'total_chunks': 'N/A', 'embedding_dimension': 'N/A'}
        
        if verbose:
            info_str = f"{{vectors: {index_info.get('total_vectors', 'N/A')}, " \
                      f"chunks: {index_info.get('total_chunks', 'N/A')}, " \
                      f"dimensions: {index_info.get('embedding_dimension', 'N/A')}}}"
            print(f"✅ Index loaded: {info_str}")
            
    def _build_index(self, verbose: bool = True) -> None:
        """Build the FAISS index using integrated logic."""
        try:
            if verbose:
                print("🔨 Building knowledge index...")
                
            # Import build logic
            from pathlib import Path
            import time
            
            # Use the same pipeline components
            input_dirs = self.config_manager.get('paths.input_dirs', ["env_scripts"])
            all_code_blocks = []
            
            # Step 1: Parse files from all input directories
            total_files = 0
            for input_dir_name in input_dirs:
                input_dir = Path(input_dir_name)
                if not input_dir.exists():
                    if verbose:
                        print(f"⚠️ Input directory not found: {input_dir}")
                    continue
                    
                files = list(input_dir.glob("*.nvn"))
                total_files += len(files)
                
                for file_path in files:
                    try:
                        blocks = self.parser.parse_file(str(file_path))
                        all_code_blocks.extend(blocks)
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error parsing {file_path.name}: {e}")
                        
            if verbose:
                print(f"• Parsed {total_files} files → {len(all_code_blocks)} blocks")
                
            # Step 2: Chunk blocks
            chunks = self.chunker.chunk_blocks(all_code_blocks)
            
            if verbose:
                print(f"• Created {len(chunks)} chunks")
                
            # Step 3: Generate embeddings
            embeddings = self.embedder.embed_chunks(chunks)
            
            if verbose:
                print(f"• Generated {embeddings.shape[0]} embeddings")
                
            # Step 4: Build and save index
            self.retriever.add_chunks(chunks, embeddings)
            
            # Save to disk
            index_path = Path("data/faiss_index")
            index_path.mkdir(parents=True, exist_ok=True)
            self.retriever.save_index(str(index_path))
            
            if verbose:
                print("✅ Index built and saved successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to build index: {e}")
            
    def query(self, question: str, transparent: bool = False) -> str:
        """
        Process a query and return the response.
        
        Args:
            question: The question to ask
            transparent: If True, show detailed processing steps
            
        Returns:
            The AI agent's response
        """
        if transparent:
            return self._query_transparent(question)
        else:
            return self._query_simple(question)
            
    def _query_simple(self, question: str) -> str:
        """Process query without showing internal steps."""
        try:
            # Convert question to embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Retrieve relevant chunks
            results = self.retriever.search(query_embedding, top_k=5)
            
            # Prepare context from results
            context = self._prepare_context_from_results(results)
            
            # Query agent
            response = self.agent.generate_response(question, context)
            
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
            
    def _query_transparent(self, question: str) -> str:
        """Process query with full transparency and logging."""
        print(f"\n🔍 PROCESSING QUERY")
        print("=" * 60)
        print(f"📝 Query: {question}")
        print(f"🤖 Agent: {self.config_manager.get_default_agent()}")
        print(f"📊 Top-K chunks: 5")
        print("-" * 60)
        
        # Create session
        session = QuerySession(question)
        start_time = time.time()
        
        try:
            # Step 1: Create query embedding
            embed_start = time.time()
            query_embedding = self.embedder.embed_text(question)
            embed_time = time.time() - embed_start
            
            session.add_step("embedding", {
                "dimensions": query_embedding.shape
            }, embed_time)
            
            print(f"🔤 Query embedding: {query_embedding.shape} dimensions ({embed_time:.3f}s)")
            
            # Step 2: Retrieve relevant chunks
            retrieve_start = time.time()
            results = self.retriever.search(query_embedding, top_k=5)
            retrieve_time = time.time() - retrieve_start
            
            session.retrieved_chunks = [result.to_dict() for result in results]
            session.timing["retrieval"] = retrieve_time
            session.add_step("retrieval", {
                "count": len(results),
                "top_scores": [r.score for r in results[:3]]
            }, retrieve_time)
            
            print(f"🔍 Retrieved {len(results)} chunks in {retrieve_time:.3f}s")
            
            # Step 2: Display retrieved chunks
            self._display_retrieved_results(results)
            
            # Step 3: Prepare context
            context_start = time.time()
            context = self._prepare_context_from_results(results)
            context_time = time.time() - context_start
            
            session.context_added = context
            session.timing["context_preparation"] = context_time
            session.add_step("context_prepared", {
                "selected_count": len(results),
                "context_length": len(context)
            }, context_time)
            
            print(f"📝 Prepared context from {len(results)} chunks in {context_time:.3f}s")
            
            # Step 4: Query LLM
            llm_start = time.time()
            llm_input = self._build_llm_prompt(question, context)
            
            session.llm_input = llm_input
            
            print(f"🤖 Querying {self.agent.__class__.__name__}...")
            print(f"📄 Input prompt length: {len(llm_input)} characters")
            
            llm_response = self.agent.generate_response(question, context)
            llm_time = time.time() - llm_start
            
            session.llm_response = llm_response
            session.timing["llm_query"] = llm_time
            session.add_step("llm_response", {
                "agent": self.agent.__class__.__name__,
                "input_length": len(llm_input),
                "response_length": len(llm_response)
            }, llm_time)
            
            print(f"💬 LLM responded in {llm_time:.3f}s")
            
            # Step 5: Display results
            total_time = time.time() - start_time
            session.timing["total"] = total_time
            
            self._display_final_results(session, llm_response)
            
            # Store session
            self.sessions.append(session)
            
            return llm_response
            
        except Exception as e:
            error_time = time.time() - start_time
            session.add_step("error", {"error": str(e)}, error_time)
            print(f"❌ Query processing failed: {e}")
            raise
            
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks (legacy)."""
        if not chunks:
            return ""
            
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0)
            
            context_parts.append(f"[Context {i} - Score: {score:.3f}]")
            context_parts.append(content)
            context_parts.append(f"Metadata: {metadata}")
            context_parts.append("---")
            
        return "\n".join(context_parts)
        
    def _prepare_context_from_results(self, results) -> str:
        """Prepare context from retrieval results."""
        if not results:
            return ""
            
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.chunk.content if hasattr(result.chunk, 'content') else str(result.chunk)
            score = result.score
            
            context_parts.append(f"[Context {i} - Score: {score:.3f}]")
            context_parts.append(content)
            context_parts.append("---")
            
        return "\n".join(context_parts)
        
    def _build_llm_prompt(self, query: str, context: str) -> str:
        """Build the complete prompt for the LLM."""
        prompt = f"""You are an expert in Envision code analysis. Your task is to answer questions about the provided code.

RELEVANT CONTEXT:
{context}

QUESTION: {query}

Please provide a detailed and accurate answer based on the context provided. If the information is not sufficient, clearly state what is missing."""
        
        return prompt
        
    def _display_retrieved_chunks(self, chunks: List[Dict]) -> None:
        """Display retrieved chunks analysis (legacy)."""
        print(f"\n📋 RETRIEVED CHUNKS ANALYSIS:")
        print("=" * 80)
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0)
            
            # Truncate content for display
            
    def _display_retrieved_results(self, results) -> None:
        """Display retrieved results analysis."""
        print(f"\n📋 RETRIEVED CHUNKS ANALYSIS:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            content = result.chunk.content if hasattr(result.chunk, 'content') else str(result.chunk)
            score = result.score
            
            # Truncate content for display
            display_content = content[:100] + "..." if len(content) > 100 else content
            
            print(f"[{i}] Score: {score:.4f}")
            print(f"    Content: {display_content}")
            if hasattr(result.chunk, 'metadata'):
                print(f"    Metadata: {result.chunk.metadata}")
            print("-" * 80)
            
    def _display_final_results(self, session: QuerySession, response: str) -> None:
        """Display final results and timing information."""
        print(f"\n🎯 FINAL RESULTS:")
        print("=" * 80)
        print(f"📊 PROCESSING SUMMARY:")
        print(f"   • Query: {session.query}")
        print(f"   • Retrieved chunks: {len(session.retrieved_chunks)}")
        print(f"   • Context length: {len(session.llm_input)} chars")
        print(f"   • Response length: {len(response)} chars")
        
        print(f"\n⏱️ TIMING BREAKDOWN:")
        for step, timing in session.timing.items():
            print(f"   • {step}: {timing:.3f}s")
            
        print(f"\n💬 LLM RESPONSE:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        # Save session
        session_file = f"data/sessions/session_{time.strftime('%Y%m%d_%H%M%S')}.json"
        session.save_to_file(session_file)
        print(f"💾 Session saved to: {session_file}")
        
    def interactive_mode(self, verbose: bool = False) -> None:
        """Run interactive chat mode."""
        if verbose:
            print("\n💬 DSL Query System - Interactive Mode (Verbose)")
            print("Ask questions about the DSL code. Type 'quit' or 'exit' to exit.")
            print("-" * 60)
        else:
            print(f"\n💬 {self.agent.model_name} ready. Type 'quit' to exit.")
        
        while True:
            try:
                question = input("\n👤 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                    
                if not question:
                    print("Please enter a question.")
                    continue
                    
                if verbose:
                    # Verbose mode: show full processing
                    response = self.query(question, transparent=True)
                else:
                    # Clean mode: just agent response
                    print(f"\n🤖 {self.agent.model_name}: ", end="", flush=True)
                    response = self.query(question, transparent=False)
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


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