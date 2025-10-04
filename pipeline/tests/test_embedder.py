"""
Test script for the embedding pipeline.

This script tests the complete preprocessing pipeline from parsing to embedding.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing.parsers.envision_parser import EnvisionParser
from preprocessing.chunkers.semantic_chunker import SemanticChunker
from preprocessing.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from preprocessing.utils.helpers import setup_logging

def test_embedding_pipeline():
    """Test the complete embedding pipeline."""
    setup_logging(level="INFO")
    
    # Initialize components
    parser = EnvisionParser()
    chunker = SemanticChunker({
        'max_chunk_tokens': 400,
        'overlap_lines': 3
    })
    
    # Use a small, fast model for testing
    embedder = SentenceTransformerEmbedder({
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 16
    })
    
    # Test file path
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "env_scripts", "67982.nvn")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing complete embedding pipeline on: {test_file}")
    print("=" * 60)
    
    try:
        # Step 1: Parse
        print("Step 1: Parsing file...")
        blocks = parser.parse_file(test_file)
        print(f"  Parsed {len(blocks)} code blocks")
        
        # Step 2: Chunk
        print("\nStep 2: Creating semantic chunks...")
        chunks = chunker.chunk_blocks(blocks)
        print(f"  Created {len(chunks)} semantic chunks")
        
        # Step 3: Initialize embedder
        print("\nStep 3: Initializing embedder...")
        try:
            embedder.initialize()
            print(f"  Initialized {embedder.model_name}")
            print(f"  Embedding dimension: {embedder.embedding_dimension}")
            print(f"  Model info: {embedder.get_model_info()}")
        except Exception as e:
            print(f"  Failed to initialize embedder: {e}")
            print("  This is expected if sentence-transformers is not installed")
            print("  Install with: pip install sentence-transformers")
            return
        
        # Step 4: Generate embeddings
        print("\nStep 4: Generating embeddings...")
        
        # Test with a smaller subset for speed
        test_chunks = chunks[:10] if len(chunks) > 10 else chunks
        print(f"  Using {len(test_chunks)} chunks for testing")
        
        embeddings = embedder.embed_chunks(test_chunks)
        print(f"  Generated embeddings shape: {embeddings.shape}")
        
        # Step 5: Analyze embeddings
        print("\nStep 5: Analyzing embeddings...")
        
        # Check embedding properties
        import numpy as np
        
        # Check for NaN or infinite values
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        # Check normalization (if enabled)
        if embedder.normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            is_normalized = np.allclose(norms, 1.0, rtol=1e-3)
            print(f"  Properly normalized: {is_normalized}")
            print(f"  Norm range: {norms.min():.3f} - {norms.max():.3f}")
        
        # Compute similarity matrix for first few chunks
        if len(test_chunks) >= 3:
            print("\n  Similarity analysis (first 3 chunks):")
            for i in range(3):
                for j in range(i+1, 3):
                    similarity = embedder.compute_similarity(embeddings[i], embeddings[j])
                    chunk_i_type = test_chunks[i].chunk_type
                    chunk_j_type = test_chunks[j].chunk_type
                    print(f"    {chunk_i_type} <-> {chunk_j_type}: {similarity:.3f}")
        
        # Step 6: Test individual text embedding
        print("\nStep 6: Testing query embedding...")
        test_query = "How to calculate demand forecasting for inventory items?"
        query_embedding = embedder.embed_text(test_query)
        print(f"  Query embedding shape: {query_embedding.shape}")
        
        # Find most similar chunk to query
        similarities = []
        for i, chunk_embedding in enumerate(embeddings):
            similarity = embedder.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, i, test_chunks[i]))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        print(f"\n  Most similar chunks to query:")
        for similarity, idx, chunk in similarities[:3]:
            chunk_preview = chunk.content[:100].replace('\n', ' ')
            print(f"    Score: {similarity:.3f} | {chunk.chunk_type} | {chunk_preview}...")
        
        print("\n✅ Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding_pipeline()