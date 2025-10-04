"""
Test script for the semantic chunker.

This script tests the chunking capabilities on parsed code blocks.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing.parsers.envision_parser import EnvisionParser
from preprocessing.chunkers.semantic_chunker import SemanticChunker
from preprocessing.utils.helpers import setup_logging

def test_semantic_chunker():
    """Test the semantic chunker on parsed code blocks."""
    setup_logging(level="INFO")
    
    # Initialize parser and chunker
    parser = EnvisionParser()
    chunker = SemanticChunker({
        'max_chunk_tokens': 300,  # Smaller for testing
        'overlap_lines': 2,
        'group_by_section': True,
        'group_related_assignments': True
    })
    
    # Test file path
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "env_scripts", "67982.nvn")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing semantic chunker on: {test_file}")
    print("=" * 60)
    
    try:
        # Parse the file first
        print("Parsing file...")
        blocks = parser.parse_file(test_file)
        print(f"Parsed {len(blocks)} code blocks")
        
        # Chunk the blocks
        print("\nChunking blocks...")
        chunks = chunker.chunk_blocks(blocks)
        print(f"Created {len(chunks)} semantic chunks")
        print()
        
        # Analyze chunks
        total_tokens = sum(chunk.size_tokens for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print(f"Chunk Statistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")
        print()
        
        # Group chunks by type
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = []
            chunk_types[chunk_type].append(chunk)
        
        print("Chunk Types:")
        for chunk_type, type_chunks in chunk_types.items():
            avg_tokens_type = sum(c.size_tokens for c in type_chunks) / len(type_chunks)
            print(f"  {chunk_type.replace('_', ' ').title()}: {len(type_chunks)} chunks (avg {avg_tokens_type:.1f} tokens)")
        print()
        
        # Show detailed analysis of first few chunks
        print("Detailed Chunk Analysis:")
        print("-" * 40)
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"Chunk {i+1}: {chunk.chunk_type}")
            print(f"  Tokens: {chunk.size_tokens}")
            print(f"  Original blocks: {len(chunk.original_blocks)}")
            print(f"  Context: {chunk.context}")
            
            # Show original block types
            block_types = [block.block_type for block in chunk.original_blocks]
            block_type_counts = {}
            for bt in block_types:
                block_type_counts[bt] = block_type_counts.get(bt, 0) + 1
            print(f"  Block composition: {dict(block_type_counts)}")
            
            # Show content preview
            content_preview = chunk.content[:200].replace('\n', ' ')
            print(f"  Content preview: {content_preview}...")
            
            # Show metadata
            if chunk.metadata:
                print(f"  Metadata: {chunk.metadata}")
            
            print()
        
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks")
        
        # Test chunk validation
        print("\nValidation Results:")
        is_valid = chunker.validate_chunks(chunks)
        print(f"All chunks valid: {is_valid}")
        
        # Check for oversized chunks
        oversized = [chunk for chunk in chunks if chunk.size_tokens > chunker.max_chunk_tokens]
        if oversized:
            print(f"Warning: {len(oversized)} chunks exceed token limit")
            for chunk in oversized:
                print(f"  - {chunk.chunk_type}: {chunk.size_tokens} tokens")
        
    except Exception as e:
        print(f"Error during chunking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_chunker()