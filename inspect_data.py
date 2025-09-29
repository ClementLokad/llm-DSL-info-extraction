"""
Script to inspect the processed data.
"""
import pickle
from pathlib import Path

def inspect_metadata():
    metadata_path = Path('processed_data/gemini_index.metadata')
    
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Embedding dimension: {data['embedding_dim']}")
    print(f"Total chunks: {len(data['metadata_list'])}")
    print("\nSample of first 3 chunks metadata:")
    
    for i, (metadata, text) in enumerate(zip(data['metadata_list'][:3], data['texts'][:3])):
        print(f"\nChunk {i+1}:")
        print("Metadata:")
        for key, value in sorted(metadata.items()):
            if isinstance(value, (list, dict)):
                print(f"  {key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    for v in value:
                        print(f"    - {v}")
            else:
                print(f"  {key}: {value}")
        print("\nText preview (first 150 chars):")
        print(f"  {text[:150]}...")
        print(f"  Length: {len(text)} characters")

if __name__ == '__main__':
    inspect_metadata()