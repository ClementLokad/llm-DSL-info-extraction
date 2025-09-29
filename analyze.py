"""
Analysis script for processed Envision data.
"""
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List

def load_results(file_path: str) -> Dict[str, Any]:
    """Load processed results from file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def print_statistics(results: Dict[str, Any]) -> None:
    """Print overall statistics."""
    print("="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    
    print(f"Files processed: {results.get('num_files', 'N/A')}")
    print(f"Total chunks: {results.get('total_chunks', 'N/A')}")
    print(f"Embedder used: {results.get('embedder_type', 'N/A')}")
    
    chunks = results.get('chunks', [])
    if chunks:
        chunk_sizes = [len(chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)) for chunk in chunks]
        print(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters")
        print(f"Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")

def analyze_content_patterns(results: Dict[str, Any]) -> None:
    """Analyze Envision code patterns."""
    print("\n" + "="*60)
    print("ENVISION CODE PATTERNS")
    print("="*60)
    
    chunks = results.get('chunks', [])
    if not chunks:
        print("No chunks to analyze.")
        return
    
    patterns = {
        'read_statements': 0,
        'const_declarations': 0,
        'export_statements': 0,
        'table_definitions': 0,
        'calculations': 0
    }
    
    for chunk in chunks:
        chunk_content = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
        chunk_lower = chunk_content.lower()
        if 'read ' in chunk_lower:
            patterns['read_statements'] += 1
        if 'const ' in chunk_lower:
            patterns['const_declarations'] += 1
        if 'export ' in chunk_lower:
            patterns['export_statements'] += 1
        if ' with ' in chunk_lower and ':' in chunk_content:
            patterns['table_definitions'] += 1
        if any(op in chunk_content for op in ['+', '-', '*', '/', '=']):
            patterns['calculations'] += 1
    
    print("Pattern distribution:")
    for pattern, count in patterns.items():
        percentage = (count / len(chunks)) * 100
        print(f"- {pattern.replace('_', ' ').title()}: {count} chunks ({percentage:.1f}%)")

def show_sample_chunks(results: Dict[str, Any], num_samples: int = 3) -> None:
    """Show sample chunks."""
    print(f"\n" + "="*60)
    print(f"SAMPLE CHUNKS (showing {num_samples})")
    print("="*60)
    
    chunks = results.get('chunks', [])
    metadata = results.get('metadata', [])
    
    for i in range(min(num_samples, len(chunks))):
        chunk = chunks[i]
        chunk_content = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
        file_path = chunk.get('file_path', 'N/A') if isinstance(chunk, dict) else 'N/A'
        
        print(f"\nChunk {i+1}:")
        print(f"  File: {file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]}")
        print(f"  Size: {len(chunk_content)} characters")
        print(f"  Content preview:")
        print(f"    {repr(chunk_content[:150])}...")

def test_search(results: Dict[str, Any], queries: List[str]) -> None:
    """Test search functionality."""
    print(f"\n" + "="*60)
    print("SEARCH FUNCTIONALITY TEST")
    print("="*60)
    
    search_engine = results.get('search_engine')
    if not search_engine:
        print("No search engine found in results.")
        return
    
    # Create a simple mock embedder for search testing
    def mock_embed(text: str) -> List[float]:
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        embedding = []
        for i in range(0, min(len(hash_hex), 768 // 4), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0
            embedding.extend([val] * 4)
        
        while len(embedding) < 768:
            embedding.append(0.0)
        
        return embedding[:768]
    
    print("Testing search with sample queries:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        try:
            query_embedding = mock_embed(query)
            search_results = search_engine.search(query_embedding, k=2)
            
            print(f"  Found {len(search_results)} results:")
            for j, result in enumerate(search_results, 1):
                print(f"    {j}. File: {result.get('file_name', 'N/A')}")
                print(f"       Distance: {result.get('distance', 'N/A'):.4f}")
                content = result.get('text', '')[:80].replace('\n', ' ')
                print(f"       Preview: {content}...")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze processed Envision data')
    parser.add_argument('--file', type=str, default='processed_data/results_mock.pkl',
                       help='Path to results file')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of sample chunks to show')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading results from: {args.file}")
        results = load_results(args.file)
        
        # Run analysis
        print_statistics(results)
        analyze_content_patterns(results)
        show_sample_chunks(results, args.samples)
        
        # Test search with common queries
        test_queries = [
            "read catalog",
            "const declaration", 
            "price calculation",
            "table definition"
        ]
        test_search(results, test_queries)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()