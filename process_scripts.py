"""
Script to process DSL files and create searchable embeddings.
"""
import os
from pathlib import Path
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

from preprocessing.processor import DSLProcessor
from preprocessing.config import registry
from preprocessing.search import DSLSearchEngine

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")
genai.configure(api_key=GEMINI_API_KEY)

def process_files(model_type: str, model_name: str, script_dir: str, storage_dir: str):
    """Process DSL files using specified model configuration."""
    # Initialize processor
    processor = DSLProcessor(
        model_type=model_type,
        model_name=model_name,
        storage_dir=storage_dir
    )
    
    # Process all DSL files
    script_dir = Path(script_dir)
    results = processor.process_directory(script_dir)
    
    # Generate embeddings for all chunks
    all_chunks = []
    for file_chunks in results.values():
        all_chunks.extend(file_chunks)
    
    print(f"Processing {len(all_chunks)} chunks from {len(results)} files...")
    
    # Generate embeddings
    embedding_data = processor.generate_embeddings(all_chunks)
    
    # Add to embedding manager
    processor.embedding_manager.add_embeddings(embedding_data)
    
    # Save processed data
    processor.save_processed_data()
    print("Processing complete! Data saved to processed_data directory.")
    
    # Print some statistics
    total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in all_chunks)
    print(f"\nStatistics:")
    print(f"Total files processed: {len(results)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Estimated total tokens: {total_tokens}")
    
    return processor

def main():
    parser = argparse.ArgumentParser(description='Process DSL files and create searchable embeddings')
    parser.add_argument('--model-type', type=str, default='gemini',
                      help='Type of model to use (e.g., gemini, gpt, mistral)')
    parser.add_argument('--model-name', type=str, default='gemini-pro',
                      help='Specific model name to use')
    parser.add_argument('--script-dir', type=str, default='env_scripts',
                      help='Directory containing DSL scripts')
    parser.add_argument('--storage-dir', type=str, default='processed_data',
                      help='Directory for storing processed data')
    parser.add_argument('--action', type=str, choices=['process', 'search', 'stats'],
                      default='process',
                      help='Action to perform')
    parser.add_argument('--query', type=str,
                      help='Search query (required for search action)')
    parser.add_argument('--filter', type=str, nargs='*',
                      help='Metadata filters in key=value format')
    
    args = parser.parse_args()
    
    # Set active model configuration
    registry.set_active_config(args.model_type, args.model_name)
    
    if args.action == 'process':
        process_files(args.model_type, args.model_name, args.script_dir, args.storage_dir)
    
    elif args.action == 'search':
        if not args.query:
            parser.error("--query is required for search action")
        
        # Parse filters
        filters = {}
        if args.filter:
            for f in args.filter:
                key, value = f.split('=', 1)
                filters[key] = value
        
        # Initialize search engine
        search_engine = DSLSearchEngine(args.storage_dir)
        
        # Perform search
        results = search_engine.search(args.query, filters=filters)
        
        # Display results
        print(f"\nSearch results for: {args.query}")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (score: {result.score:.3f}):")
            print(f"File: {result.metadata['file_path']}")
            print(f"Lines: {result.metadata['start_line']}-{result.metadata['end_line']}")
            if 'dsl_elements' in result.metadata:
                print(f"DSL elements: {', '.join(result.metadata['dsl_elements'])}")
            print("\nPreview:")
            preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
            print(preview)
            print("-" * 80)
    
    elif args.action == 'stats':
        search_engine = DSLSearchEngine(args.storage_dir)
        stats = search_engine.get_stats()
        
        print("\nIndex Statistics:")
        for model_id, model_stats in stats.items():
            print(f"\n{model_id}:")
            for key, value in model_stats.items():
                print(f"  {key}: {value}")

if __name__ == '__main__':
    main()