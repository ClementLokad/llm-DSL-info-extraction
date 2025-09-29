#!/usr/bin/env python3
"""
Example script showing the modular embedder system.
"""

from agents import get_embedder, list_available_embedders

def main():
    print("🤖 LOKAD Embedder System Demo")
    print("=" * 50)
    
    # List available embedders
    available = list_available_embedders()
    print(f"Available embedders: {', '.join(available)}")
    print()
    
    # Test mock embedder
    print("Testing Mock Embedder:")
    mock_embedder = get_embedder('mock')
    
    sample_text = "const totalSales = sum(Sales.Amount)"
    embedding = mock_embedder.embed(sample_text)
    
    print(f"  Text: {sample_text}")
    print(f"  Dimension: {mock_embedder.dimension}")
    print(f"  Embedding sample: {embedding[:5]}... (first 5 values)")
    print()
    
    print("✅ Demo complete! The system is fully modular and extensible.")

if __name__ == '__main__':
    main()