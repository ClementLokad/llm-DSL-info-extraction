#!/usr/bin/env python3
"""
Configuration demonstration script.

This script shows how to access and use all configuration parameters
from both .env (secrets) and config.yaml (application settings).
"""

import os
from config_manager import get_config

def main():
    """Demonstrate configuration usage."""
    print("🔧 Configuration System Demonstration")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    print("Environment: dev (default)")
    print()
    
    # API Keys (from .env)
    print("🔐 API Keys (from .env):")
    print(f"  OpenAI: {'✅ Set' if config.get_api_key('openai') else '❌ Missing'}")
    print(f"  Google: {'✅ Set' if config.get_api_key('google') else '❌ Missing'}")
    print(f"  Mistral: {'✅ Set' if config.get_api_key('mistral') else '❌ Missing'}")
    print()
    
    # Parser Configuration (from YAML)
    print("🔍 Parser Configuration (from config.yaml):")
    parser_config = config.get_parser_config()
    print(f"  Type: {parser_config.get('type')}")
    print(f"  Extensions: {parser_config.get('supported_extensions')}")
    section_config = parser_config.get('section_delimiter', {})
    print(f"  Section delimiter min chars: {section_config.get('min_chars')}")
    print(f"  Section delimiter valid chars: {section_config.get('valid_chars')}")
    print()
    
    # Chunker Configuration (from YAML)
    print("🧩 Chunker Configuration (from config.yaml):")
    chunker_config = config.get_chunker_config()
    print(f"  Type: {chunker_config.get('type')}")
    print(f"  Max chunk tokens: {chunker_config.get('max_chunk_tokens')}")
    print(f"  Overlap lines: {chunker_config.get('overlap_lines')}")
    print(f"  Preserve boundaries: {chunker_config.get('preserve_boundaries')}")
    print(f"  Chars per token: {chunker_config.get('chars_per_token')}")
    
    strategies = chunker_config.get('strategies', {})
    print("  Strategies:")
    for key, value in strategies.items():
        print(f"    {key}: {value}")
    print()
    
    # Embedder Configuration (from YAML)
    print("🧠 Embedder Configuration (from config.yaml):")
    embedder_config = config.get_embedder_config()
    print(f"  Default type: {embedder_config.get('type')}")
    
    # Show specific embedder configs
    for embedder_type in ['sentence_transformer', 'openai', 'gemini']:
        specific_config = config.get_embedder_config(embedder_type)
        print(f"  {embedder_type.title()}:")
        for key, value in specific_config.get('specific', {}).items():
            print(f"    {key}: {value}")
    print()
    
    # Retriever Configuration (from YAML)
    print("🔍 Retriever Configuration (from config.yaml):")
    retriever_config = config.get_retriever_config()
    print(f"  Type: {retriever_config.get('type')}")
    faiss_config = retriever_config.get('faiss', {})
    print(f"  FAISS index type: {faiss_config.get('default_index_type')}")
    print(f"  FAISS top_k: {faiss_config.get('top_k')}")
    print(f"  FAISS use_gpu: {faiss_config.get('use_gpu')}")
    print()
    
    # Pipeline Configuration (from YAML)
    print("⚙️ Pipeline Configuration (from config.yaml):")
    pipeline_config = config.get_pipeline_config()
    print(f"  Batch size: {pipeline_config.get('batch_size')}")
    print(f"  Max workers: {pipeline_config.get('max_workers')}")
    print(f"  Supported extensions: {pipeline_config.get('supported_extensions')}")
    print(f"  Encoding: {pipeline_config.get('encoding')}")
    print()
    
    print()
    print("🎉 Configuration demonstration complete!")
    print()
    print("💡 Tips:")
    print("  - Edit .env for secrets (API keys, paths)")
    print("  - Edit config.yaml for application parameters")
    print("  - All numeric constants come from config files")
    print("  - No hardcoded values in Python code!")

if __name__ == "__main__":
    main()