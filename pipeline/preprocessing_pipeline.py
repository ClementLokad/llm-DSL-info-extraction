"""
Main preprocessing pipeline for Envision DSL codebase analysis.

This script provides a complete preprocessing pipeline from parsing to retrieval.
"""

import os
import argparse
from typing import List, Optional, Dict, Any
import logging

from preprocessing.parsers.envision_parser import EnvisionParser
from preprocessing.chunkers.semantic_chunker import SemanticChunker
from preprocessing.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from preprocessing.embedders.openai_embedder import OpenAIEmbedder
from preprocessing.embedders.gemini_embedder import GeminiEmbedder
from preprocessing.retrievers.faiss_retriever import FAISSRetriever
from preprocessing.utils.helpers import setup_logging, time_function, create_progress_callback
from preprocessing.utils.config import ConfigManager

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Envision DSL codebase analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration dictionary or None for defaults
        """
        # Load configuration
        if isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config_manager.update(config)
        elif isinstance(config, ConfigManager):
            self.config_manager = config
        else:
            self.config_manager = ConfigManager.create_default_config()
        
        # Initialize components
        self.parser = None
        self.chunker = None
        self.embedder = None
        self.retriever = None
        
        # Setup logging
        log_level = self.config_manager.get('logging.level', 'INFO')
        setup_logging(level=log_level)
    
    @time_function
    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing preprocessing pipeline components...")
        
        # Initialize parser
        parser_config = self.config_manager.get_section('parser')
        self.parser = EnvisionParser(parser_config)
        logger.info(f"Initialized parser: {type(self.parser).__name__}")
        
        # Initialize chunker
        chunker_config = self.config_manager.get_section('chunker')
        self.chunker = SemanticChunker(chunker_config)
        logger.info(f"Initialized chunker: {type(self.chunker).__name__}")
        
        # Initialize embedder
        embedder_type = self.config_manager.get('embedder.type', 'sentence_transformer')
        embedder_config = self.config_manager.get_section('embedder')
        
        if embedder_type == 'sentence_transformer':
            self.embedder = SentenceTransformerEmbedder(embedder_config)
        elif embedder_type == 'openai':
            self.embedder = OpenAIEmbedder(embedder_config)
        elif embedder_type == 'gemini':
            self.embedder = GeminiEmbedder(embedder_config)
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
        
        self.embedder.initialize()
        logger.info(f"Initialized embedder: {type(self.embedder).__name__}")
        
        # Initialize retriever
        retriever_config = self.config_manager.get_section('retriever')
        self.retriever = FAISSRetriever(retriever_config)
        self.retriever.initialize(self.embedder.embedding_dimension)
        logger.info(f"Initialized retriever: {type(self.retriever).__name__}")
    
    @time_function
    def process_files(self, file_paths: List[str], save_index: bool = True) -> None:
        """
        Process a list of Envision files through the complete pipeline.
        
        Args:
            file_paths: List of paths to Envision files
            save_index: Whether to save the index after processing
        """
        if not self.parser or not self.chunker or not self.embedder or not self.retriever:
            raise RuntimeError("Components not initialized. Call initialize_components() first.")
        
        logger.info(f"Processing {len(file_paths)} files...")
        
        all_chunks = []
        progress_callback = create_progress_callback(len(file_paths), "Processing files")
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            try:
                logger.debug(f"Processing file: {file_path}")
                
                # Parse file
                blocks = self.parser.parse_file(file_path)
                logger.debug(f"  Parsed {len(blocks)} blocks")
                
                # Chunk blocks
                chunks = self.chunker.chunk_blocks(blocks)
                logger.debug(f"  Created {len(chunks)} chunks")
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
            
            progress_callback(i + 1)
        
        if not all_chunks:
            logger.warning("No chunks created from input files")
            return
        
        logger.info(f"Created {len(all_chunks)} total chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_chunks(all_chunks)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Add to retriever
        logger.info("Adding chunks to retriever...")
        self.retriever.add_chunks(all_chunks, embeddings)
        
        # Save index if requested
        if save_index:
            logger.info("Saving index...")
            self.retriever.save_index()
            logger.info("Index saved successfully")
    
    def search(self, query: str, top_k: int = 10) -> List:
        """
        Search for relevant code chunks using natural language query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        if not self.retriever or not self.embedder:
            raise RuntimeError("Pipeline not initialized or no index loaded")
        
        return self.retriever.search_by_text(query, self.embedder, top_k)
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        Load a saved index.
        
        Args:
            index_path: Path to load index from (uses config default if None)
        """
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")
        
        self.retriever.load_index(index_path)
        logger.info("Index loaded successfully")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed data.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'pipeline_initialized': all([
                self.parser is not None,
                self.chunker is not None,
                self.embedder is not None,
                self.retriever is not None
            ])
        }
        
        if self.retriever:
            stats.update(self.retriever.get_statistics())
        
        if self.embedder and hasattr(self.embedder, 'get_model_info'):
            stats['embedder_info'] = self.embedder.get_model_info()
        
        return stats

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Envision DSL Preprocessing Pipeline")
    parser.add_argument('action', choices=['process', 'search', 'test'], 
                       help='Action to perform')
    parser.add_argument('--input', '-i', help='Input file or directory')
    parser.add_argument('--query', '-q', help='Search query')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output directory for index')
    parser.add_argument('--top-k', '-k', type=int, default=10, 
                       help='Number of search results')
    parser.add_argument('--embedder', choices=['sentence_transformer', 'openai', 'gemini'],
                       default='sentence_transformer', help='Embedder type')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    
    # Load configuration
    config_manager = ConfigManager.create_default_config()
    if args.config:
        config_manager.load_config(args.config)
    
    # Override config with command line arguments
    if args.output:
        config_manager.set('retriever.index_path', args.output)
    if args.embedder:
        config_manager.set('embedder.type', args.embedder)
    
    # Create pipeline
    pipeline = PreprocessingPipeline(config_manager)
    
    try:
        if args.action == 'process':
            if not args.input:
                print("Input file or directory required for processing")
                return
            
            # Initialize components
            pipeline.initialize_components()
            
            # Get file paths
            if os.path.isfile(args.input):
                file_paths = [args.input]
            elif os.path.isdir(args.input):
                file_paths = []
                for root, dirs, files in os.walk(args.input):
                    for file in files:
                        if file.endswith('.nvn'):
                            file_paths.append(os.path.join(root, file))
            else:
                print(f"Input path not found: {args.input}")
                return
            
            if not file_paths:
                print("No .nvn files found")
                return
            
            # Process files
            pipeline.process_files(file_paths)
            
            # Print statistics
            stats = pipeline.get_statistics()
            print("\nProcessing completed!")
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Index path: {config_manager.get('retriever.index_path')}")
            
        elif args.action == 'search':
            if not args.query:
                print("Query required for search")
                return
            
            # Initialize components
            pipeline.initialize_components()
            
            # Load index
            pipeline.load_index()
            
            # Perform search
            results = pipeline.search(args.query, args.top_k)
            
            print(f"\nSearch results for: {args.query}")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                chunk = result.chunk
                print(f"{i}. Score: {result.score:.3f}")
                print(f"   Type: {chunk.chunk_type}")
                print(f"   Context: {chunk.context}")
                preview = chunk.content[:200].replace('\n', ' ')
                print(f"   Preview: {preview}...")
                print()
            
        elif args.action == 'test':
            # Run a simple test
            print("Running pipeline test...")
            
            # Find test file
            test_file = os.path.join(os.path.dirname(__file__), 
                                   "..", "env_scripts", "67982.nvn")
            
            if not os.path.exists(test_file):
                print(f"Test file not found: {test_file}")
                return
            
            # Initialize and process
            pipeline.initialize_components()
            pipeline.process_files([test_file], save_index=False)
            
            # Test search
            test_query = "demand forecasting calculation"
            results = pipeline.search(test_query, top_k=3)
            
            print(f"\nTest search results for: {test_query}")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.chunk.chunk_type} (score: {result.score:.3f})")
            
            print("✅ Test completed successfully!")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()