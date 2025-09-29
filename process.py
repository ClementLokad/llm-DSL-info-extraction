"""
Main script for processing Envision DSL files.
Modular design allows easy addition of new embedding models.
"""
import argparse
import os
from pathlib import Path
from preprocessing.pipeline import PreprocessingPipeline
from agents import get_embedder, list_available_embedders

def create_embedder(model_type: str, api_key: str = ""):
    """Factory function to create embedder using the modular system."""
    # Prepare arguments based on model type
    kwargs = {}
    
    if model_type == "gemini":
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY', '')
        if not api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or use --api-key.")
        kwargs['api_key'] = api_key
    
    elif model_type == "openai":
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key.")
        kwargs['api_key'] = api_key
    
    elif model_type == "mock":
        # Mock embedder doesn't need API key
        pass
    
    # Use the modular embedder system
    try:
        return get_embedder(model_type, **kwargs)
    except ValueError as e:
        available = list_available_embedders()
        raise ValueError(f"{e}. Available embedders: {available}")

def main():
    # Load environment variables first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get available embedders dynamically
    available_embedders = list_available_embedders()
    
    # Get defaults from environment
    default_model = os.getenv('DEFAULT_EMBEDDING_MODEL', 'mock')
    default_chunk_size = int(os.getenv('DEFAULT_CHUNK_SIZE', '512'))
    default_overlap = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '50'))
    
    # Ensure default model is available
    if default_model not in available_embedders:
        default_model = 'mock'
    
    parser = argparse.ArgumentParser(description='Process Envision files')
    parser.add_argument('--script-dir', default='env_scripts', help='Directory with Envision scripts')
    parser.add_argument('--output-dir', default='processed_data', help='Output directory')
    parser.add_argument('--model', choices=available_embedders, default=default_model, 
                       help=f'Model type. Available: {", ".join(available_embedders)} (default: {default_model})')
    parser.add_argument('--api-key', help='API key for the model (overrides environment variables)')
    parser.add_argument('--chunk-size', type=int, default=default_chunk_size, help=f'Chunk size (default: {default_chunk_size})')
    parser.add_argument('--overlap', type=int, default=default_overlap, help=f'Chunk overlap (default: {default_overlap})')
    parser.add_argument('--max-files', type=int, help='Max files to process (for testing)')
    
    args = parser.parse_args()
    
    print(f"Processing with {args.model} model")
    print(f"Available embedders: {', '.join(available_embedders)}")
    
    # Create embedder
    try:
        embedder = create_embedder(args.model, args.api_key or "")
        print(f"Using {type(embedder).__name__} (dimension: {embedder.dimension})")
    except Exception as e:
        print(f"Error creating embedder: {e}")
        return
    
    # Create pipeline
    pipeline = PreprocessingPipeline(embedder, args.chunk_size, args.overlap)
    
    # Process files
    try:
        if args.max_files:
            # Process limited files for testing
            script_path = Path(args.script_dir)
            files = list(script_path.glob("*.nvn"))[:args.max_files]
            temp_dir = Path(args.output_dir) / "temp_input"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                temp_file = temp_dir / file_path.name
                temp_file.write_text(file_path.read_text(encoding='utf-8', errors='ignore'))
            
            results = pipeline.process_files(str(temp_dir), "*.nvn")
            
            import shutil
            shutil.rmtree(temp_dir)
        else:
            results = pipeline.process_files(args.script_dir, "*.nvn")
        
        # Save results
        output_file = Path(args.output_dir) / f"results_{args.model}.pkl"
        pipeline.save_results(results, str(output_file))
        
        print(f"\nComplete! Files: {results['num_files']}, Chunks: {results['total_chunks']}")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()