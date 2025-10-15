#!/usr/bin/env python3
"""
Build search index for the DSL Query System.

This script performs the complete preprocessing pipeline:
1. Parse all source files in the data directories
2. Create semantic chunks from parsed code blocks
3. Generate embeddings for all chunks  
4. Build and save the FAISS search index

Usage:
    python build_index.py                    # Build index (check if needed)
    python build_index.py --force            # Force rebuild even if index exists
    python build_index.py --input-dir path   # Process specific directory
    python build_index.py --quiet            # Suppress detailed output
    python build_index.py --check            # Only check index status
"""

import sys
import argparse
from pathlib import Path
import time
from typing import List

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config_manager import ConfigManager
from pipeline.parsers.envision_parser import EnvisionParser
from pipeline.chunkers.semantic_chunker import SemanticChunker
from pipeline.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from pipeline.retrievers.faiss_retriever import FAISSRetriever

class IndexBuilder:
    """Builder for the complete search index."""
    
    def __init__(self, force_rebuild: bool = False, quiet: bool = False):
        """Initialize the index builder."""
        self.force_rebuild = force_rebuild
        self.quiet = quiet
        self.config_manager = ConfigManager()
        
        # Initialize components
        self.parser = None
        self.chunker = None
        self.embedder = None
        self.retriever = None
        
        self._initialize_components(quiet)
    
    def _initialize_components(self, quiet=False):
        """Initialize all pipeline components."""
        if not quiet:
            print("🔧 Initializing pipeline components...")
        
        # Parser
        parser_config = self.config_manager.get_parser_config()
        self.parser = EnvisionParser(config=parser_config)
        if not quiet:
            print(f"✅ Parser initialized: {parser_config.get('type')}")
        
        # Chunker
        chunker_config = self.config_manager.get_chunker_config()
        self.chunker = SemanticChunker(config=chunker_config)
        if not quiet:
            print(f"✅ Chunker initialized: max_tokens={chunker_config.get('max_chunk_tokens')}")
        
        # Embedder
        embedder_config = self.config_manager.get_embedder_config()
        self.embedder = SentenceTransformerEmbedder(config=embedder_config)
        self.embedder.initialize()  # Initialize the model
        if not quiet:
            print(f"✅ Embedder initialized: {embedder_config.get('model_name')}")
        
        # Retriever
        retriever_config = self.config_manager.get_retriever_config()
        self.retriever = FAISSRetriever(config=retriever_config)
        if not quiet:
            print(f"✅ Retriever initialized: {retriever_config.get('faiss', {}).get('index_type')}")
    
    def find_source_files(self, input_dirs: List[str]) -> List[Path]:
        """Find all source files to process."""
        if not self.quiet:
            print(f"🔍 Scanning for source files in: {input_dirs}")
        
        supported_extensions = self.config_manager.get_parser_config().get('supported_extensions', ['.nvn'])
        source_files = []
        
        for input_dir in input_dirs:
            dir_path = Path(input_dir)
            if not dir_path.exists():
                if not self.quiet:
                    print(f"⚠️ Directory not found: {input_dir}")
                continue
            
            for ext in supported_extensions:
                files = list(dir_path.glob(f"**/*{ext}"))
                source_files.extend(files)
                print(f"   Found {len(files)} {ext} files in {input_dir}")
        
        print(f"📁 Total files to process: {len(source_files)}")
        return source_files
    
    def process_files(self, source_files: List[Path]) -> tuple:
        """Process all source files through the pipeline."""
        print("\n🏭 Processing files through pipeline...")
        
        all_code_blocks = []
        processed_files = 0
        failed_files = 0
        
        start_time = time.time()
        
        for file_path in source_files:
            try:
                print(f"📄 Processing: {file_path.name}")
                
                # Parse file
                code_blocks = self.parser.parse_file(str(file_path))
                
                # Add source file info to blocks
                for block in code_blocks:
                    block.metadata = block.metadata or {}
                    try:
                        # Try to get relative path to project root
                        block.metadata['source_file'] = str(file_path.relative_to(project_root))
                    except ValueError:
                        # If file is not within project root, use relative to current dir
                        block.metadata['source_file'] = str(file_path.name)
                
                all_code_blocks.extend(code_blocks)
                processed_files += 1
                
                if processed_files % 10 == 0:
                    print(f"   Processed {processed_files}/{len(source_files)} files...")
                
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
                failed_files += 1
        
        parse_time = time.time() - start_time
        print(f"✅ Parsing complete: {processed_files} files, {len(all_code_blocks)} blocks in {parse_time:.2f}s")
        
        if failed_files > 0:
            print(f"⚠️ {failed_files} files failed to process")
        
        # Create semantic chunks
        print("\n🧩 Creating semantic chunks...")
        chunk_start = time.time()
        
        chunks = self.chunker.chunk_blocks(all_code_blocks)
        
        chunk_time = time.time() - chunk_start
        print(f"✅ Chunking complete: {len(chunks)} chunks in {chunk_time:.2f}s")
        
        return chunks, {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_blocks': len(all_code_blocks),
            'total_chunks': len(chunks),
            'parse_time': parse_time,
            'chunk_time': chunk_time
        }
    
    def build_index(self, chunks: List) -> dict:
        """Build the search index from chunks."""
        print("\n🎯 Generating embeddings...")
        embed_start = time.time()
        
        # Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)
        
        embed_time = time.time() - embed_start
        print(f"✅ Embeddings generated: {embeddings.shape} in {embed_time:.2f}s")
        
        # Initialize retriever with embedding dimension
        embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        self.retriever.initialize(embedding_dim)
        
        print("\n💾 Building search index...")
        index_start = time.time()
        
        # Add chunks to index
        self.retriever.add_chunks(chunks, embeddings)
        
        # Save index to disk
        self.retriever.save_index()
        
        index_time = time.time() - index_start
        print(f"✅ Index built and saved in {index_time:.2f}s")
        
        # Get index info
        index_info = self.retriever.get_index_info()
        
        return {
            'embedding_time': embed_time,
            'index_time': index_time,
            'index_info': index_info
        }
    
    def run(self, input_dirs: List[str]) -> bool:
        """Run the complete index building process."""
        total_start = time.time()
        
        print("🏗️ BUILDING SEARCH INDEX")
        print("="*50)
        
        # Check if index already exists
        if not self.force_rebuild:
            try:
                self.retriever.load_index()
                index_info = self.retriever.get_index_info()
                print(f"ℹ️ Index already exists: {index_info['total_chunks']} chunks")
                
                response = input("🤔 Rebuild anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("👍 Using existing index")
                    return True
            except:
                pass  # No existing index, continue with build
        
        try:
            # Find source files
            source_files = self.find_source_files(input_dirs)
            if not source_files:
                print("❌ No source files found to process")
                return False
            
            # Process files
            chunks, process_stats = self.process_files(source_files)
            if not chunks:
                print("❌ No chunks created from source files")
                return False
            
            # Build index
            index_stats = self.build_index(chunks)
            
            # Final summary
            total_time = time.time() - total_start
            
            print("\n" + "="*50)
            print("📊 INDEX BUILD SUMMARY")
            print("="*50)
            print(f"📁 Files processed: {process_stats['processed_files']} (failed: {process_stats['failed_files']})")
            print(f"🧱 Code blocks: {process_stats['total_blocks']}")
            print(f"🧩 Semantic chunks: {process_stats['total_chunks']}")
            print(f"🎯 Embeddings: {index_stats['index_info']['embedding_dimension']}d vectors")
            print(f"💾 Index type: {index_stats['index_info']['index_type']}")
            print(f"⏱️ Total time: {total_time:.2f}s")
            print(f"✅ Index saved to: {self.retriever.index_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Index build failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_complete_index(self, input_dirs: List[str] = None) -> bool:
        """Build complete index with default directories."""
        if input_dirs is None:
            input_dirs = self.config_manager.get('paths.input_dirs', ["env_scripts"])
        
        return self.run(input_dirs)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build search index for DSL Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_index.py                    # Build index if needed
    python build_index.py --force            # Force rebuild
    python build_index.py --check            # Check index status only
    python build_index.py --quiet            # Suppress detailed output
        """
    )
    
    parser.add_argument("--input-dir", "-i", action="append", 
                       help="Input directories to process (can be used multiple times)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force rebuild even if index exists")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output")
    parser.add_argument("--check", "-c", action="store_true",
                       help="Only check index status without building")
    
    args = parser.parse_args()
    
    # Check index status only
    if args.check:
        index_path = Path("data/faiss_index")
        if index_path.exists() and (index_path / "faiss.index").exists():
            try:
                import faiss
                import pickle
                
                index = faiss.read_index(str(index_path / "faiss.index"))
                with open(index_path / "chunks.pkl", 'rb') as f:
                    chunks = pickle.load(f)
                    
                if not args.quiet:
                    print(f"✅ Index exists and is valid:")
                    print(f"   • Location: {index_path}")
                    print(f"   • Vectors: {index.ntotal}")
                    print(f"   • Dimensions: {index.d}")
                    print(f"   • Chunks: {len(chunks)}")
                return 0
            except Exception as e:
                if not args.quiet:
                    print(f"❌ Index exists but is corrupted: {e}")
                return 1
        else:
            if not args.quiet:
                print(f"❌ Index not found at: {index_path}")
            return 1
    
    # Default input directories from config
    config_manager = ConfigManager()
    default_input_dirs = config_manager.get('paths.input_dirs', ["env_scripts"])
    input_dirs = args.input_dir or default_input_dirs
    
    # Validate input directories exist
    for input_dir in input_dirs:
        if not Path(input_dir).exists():
            if not args.quiet:
                print(f"❌ Input directory not found: {input_dir}")
            return 1
    
    # Check if rebuild is needed
    index_path = Path("data/faiss_index")
    if not args.force and index_path.exists():
        if not args.quiet:
            print(f"✅ Index already exists at: {index_path}")
            print("Use --force to rebuild or --check to verify")
        return 0
    
    # Build index
    if not args.quiet:
        print("🏗️ Building search index...")
        
    builder = IndexBuilder(force_rebuild=args.force, quiet=args.quiet)
    
    # Override verbosity for quiet mode
    if args.quiet:
        # Temporarily redirect stdout to suppress output
        import contextlib
        import io
        
        with contextlib.redirect_stdout(io.StringIO()):
            success = builder.run(input_dirs)
    else:
        success = builder.run(input_dirs)
    
    if success:
        if not args.quiet:
            print("✅ Index built successfully!")
        return 0
    else:
        if not args.quiet:
            print("❌ Index build failed!")
        return 1

if __name__ == "__main__":
    exit(main())