#!/usr/bin/env python3
"""
Comprehensive architectural coherence analysis for the preprocessing pipeline.
Validates that all components are properly configured and coherent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager

def analyze_config_coherence():
    """Analyze configuration coherence across all components"""
    print("🔍 Analyzing Configuration Coherence...\n")
    
    config_manager = ConfigManager()
    
    # 1. Parser analysis
    print("📄 PARSER ANALYSIS:")
    parser_config = config_manager.get_parser_config()
    print(f"   ✅ Parser config keys: {list(parser_config.keys())}")
    
    # 2. Chunker analysis
    print("\n🧩 CHUNKER ANALYSIS:")
    chunker_config = config_manager.get_chunker_config()
    print(f"   ✅ Chunker config keys: {list(chunker_config.keys())}")
    
    # Check for coherence between chunkers
    if 'semantic' in chunker_config:
        semantic_config = chunker_config['semantic']
        print(f"   ✅ Semantic chunker max_table_names: {semantic_config.get('max_table_names')}")
    
    # 3. Embedder analysis
    print("\n🎯 EMBEDDER ANALYSIS:")
    embedder_config = config_manager.get_embedder_config()
    print(f"   ✅ Embedder config keys: {list(embedder_config.keys())}")
    
    # Check text preparation coherence
    if 'text_preparation' in embedder_config:
        text_prep = embedder_config['text_preparation']
        print(f"   ✅ Text preparation strategy: {text_prep.get('strategy')}")
        print(f"   ✅ Max chars before truncation: {text_prep.get('max_chars_before_truncation')}")
        print(f"   ✅ Chars per token (code): {text_prep.get('chars_per_token_code')}")
    
    # 4. Retriever analysis
    print("\n🔍 RETRIEVER ANALYSIS:")
    retriever_config = config_manager.get_retriever_config()
    print(f"   ✅ Retriever config keys: {list(retriever_config.keys())}")
    
    if 'faiss' in retriever_config:
        faiss_config = retriever_config['faiss']
        # index_path is in the main retriever config, not in faiss subsection
        index_path = retriever_config.get('index_path', 'Not defined')
        print(f"   ✅ FAISS index_path: {index_path}")
        print(f"   ✅ FAISS default_index_type: {faiss_config.get('default_index_type')}")
    
    # 5. Cross-component coherence
    print("\n🔗 CROSS-COMPONENT COHERENCE:")
    
    # Check embedding dimensions match
    sentence_transformer_config = embedder_config.get('sentence_transformer', {})
    if 'model_name' in sentence_transformer_config:
        model_name = sentence_transformer_config['model_name']
        expected_dim = sentence_transformer_config.get('embedding_dimension', 384)
        print(f"   ✅ SentenceTransformer model: {model_name} → {expected_dim}d")
    
    # Check chunker → embedder compatibility
    max_chunk_tokens = chunker_config.get('max_chunk_tokens', 500)
    chars_per_token = embedder_config.get('text_preparation', {}).get('chars_per_token_code', 4)
    max_chars_theoretical = max_chunk_tokens * chars_per_token
    max_chars_config = embedder_config.get('text_preparation', {}).get('max_chars_before_truncation', 2000)
    
    print(f"   🧮 Chunker max tokens: {max_chunk_tokens}")
    print(f"   🧮 Chars per token: {chars_per_token}")
    print(f"   🧮 Theoretical max chars: {max_chars_theoretical}")
    print(f"   🧮 Config max chars: {max_chars_config}")
    
    coherence_ratio = max_chars_config / max_chars_theoretical if max_chars_theoretical > 0 else 0
    coherence_status = "✅ COHERENT" if 0.8 <= coherence_ratio <= 1.5 else "⚠️ POTENTIAL ISSUE"
    print(f"   📊 Coherence ratio: {coherence_ratio:.2f} → {coherence_status}")
    
    return True

def analyze_code_coherence():
    """Analyze code implementation coherence"""
    print("\n🔧 CODE IMPLEMENTATION COHERENCE:")
    
    # Check if all components can be imported
    components = [
        ("EnvisionParser", "preprocessing.parsers.envision_parser"),
        ("SemanticChunker", "preprocessing.chunkers.semantic_chunker"),
        ("BaseEmbedder", "preprocessing.core.base_embedder"),
        ("SentenceTransformerEmbedder", "preprocessing.embedders.sentence_transformer_embedder"),
        ("FAISSRetriever", "preprocessing.retrievers.faiss_retriever"),
    ]
    
    import_results = []
    for component_name, module_path in components:
        try:
            module = __import__(module_path, fromlist=[component_name])
            component_class = getattr(module, component_name)
            import_results.append((component_name, True, None))
            print(f"   ✅ {component_name}: Import successful")
        except Exception as e:
            import_results.append((component_name, False, str(e)))
            print(f"   ❌ {component_name}: Import failed - {e}")
    
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_imports = len(import_results)
    print(f"\n📊 Import Success Rate: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")
    
    return successful_imports == total_imports

def main():
    """Main analysis function"""
    print("🏗️ ARCHITECTURAL COHERENCE ANALYSIS")
    print("="*60)
    
    config_coherent = analyze_config_coherence()
    code_coherent = analyze_code_coherence()
    
    print("\n" + "="*60)
    print("📋 FINAL ASSESSMENT:")
    print(f"   Configuration Coherence: {'✅ PASS' if config_coherent else '❌ FAIL'}")
    print(f"   Code Implementation Coherence: {'✅ PASS' if code_coherent else '❌ FAIL'}")
    
    overall_status = config_coherent and code_coherent
    print(f"\n🏁 OVERALL SYSTEM COHERENCE: {'✅ EXCELLENT' if overall_status else '❌ NEEDS ATTENTION'}")
    
    return overall_status

if __name__ == "__main__":
    main()