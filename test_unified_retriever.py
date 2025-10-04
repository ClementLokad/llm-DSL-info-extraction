#!/usr/bin/env python3
"""
Comprehensive test for the unified retriever system.
Tests path management, configuration coherence, and HNSW parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from preprocessing.retrievers.faiss_retriever import FAISSRetriever

def test_path_unification():
    """Test that path management is unified and coherent."""
    print("🔍 Testing unified path management...")
    
    config_manager = ConfigManager()
    retriever_config = config_manager.get_retriever_config()
    
    retriever = FAISSRetriever(config=retriever_config)
    
    # Test path utility methods
    base_path = "./test_index"
    
    chunks_path = retriever._get_chunks_file_path(base_path)
    index_path = retriever._get_index_file_path(base_path)
    metadata_path = retriever._get_metadata_file_path(base_path)
    
    print(f"   ✅ Chunks path: {chunks_path}")
    print(f"   ✅ Index path: {index_path}")
    print(f"   ✅ Metadata path: {metadata_path}")
    
    # Verify consistency
    # Use the same filenames as the retriever constructs them
    expected_chunks = retriever._get_chunks_file_path(base_path)
    expected_index = retriever._get_index_file_path(base_path) 
    expected_metadata = retriever._get_metadata_file_path(base_path)
    
    assert chunks_path == expected_chunks, f"Chunks path mismatch: {chunks_path} != {expected_chunks}"
    assert index_path == expected_index, f"Index path mismatch: {index_path} != {expected_index}"
    assert metadata_path == expected_metadata, f"Metadata path mismatch: {metadata_path} != {expected_metadata}"
    
    print("   ✅ Path unification: PASS")
    return True

def test_config_coherence():
    """Test configuration coherence and parameter loading."""
    print("\n🔧 Testing configuration coherence...")
    
    config_manager = ConfigManager()
    retriever_config = config_manager.get_retriever_config()
    
    retriever = FAISSRetriever(config=retriever_config)
    
    # Test that all parameters are loaded correctly
    print(f"   ✅ Index type: {retriever.index_type}")
    print(f"   ✅ Use GPU: {retriever.use_gpu}")
    print(f"   ✅ IVF nlist: {retriever.nlist}")
    print(f"   ✅ IVF nprobe: {retriever.nprobe}")
    print(f"   ✅ HNSW M: {retriever.hnsw_m}")
    print(f"   ✅ HNSW ef_construction: {retriever.hnsw_ef_construction}")
    print(f"   ✅ HNSW ef_search: {retriever.hnsw_ef_search}")
    
    # Verify HNSW parameters are from config
    faiss_config = retriever_config.get('faiss', {})
    assert retriever.hnsw_m == faiss_config.get('m', 16), "HNSW M parameter mismatch"
    assert retriever.hnsw_ef_construction == faiss_config.get('ef_construction', 200), "HNSW ef_construction mismatch"
    assert retriever.hnsw_ef_search == faiss_config.get('ef_search', 64), "HNSW ef_search mismatch"
    
    print("   ✅ Configuration coherence: PASS")
    return True

def test_hnsw_parameters():
    """Test HNSW parameter handling."""
    print("\n⚙️ Testing HNSW parameter handling...")
    
    # Test with HNSW configuration
    hnsw_config = {
        'faiss': {
            'index_type': 'IndexHNSWFlat',
            'use_gpu': False,
            'm': 32,
            'ef_construction': 400,
            'ef_search': 128
        },
        'index_path': './test_hnsw_index'
    }
    
    retriever = FAISSRetriever(config=hnsw_config)
    
    print(f"   ✅ HNSW index type: {retriever.index_type}")
    print(f"   ✅ HNSW M: {retriever.hnsw_m}")
    print(f"   ✅ HNSW ef_construction: {retriever.hnsw_ef_construction}")
    print(f"   ✅ HNSW ef_search: {retriever.hnsw_ef_search}")
    
    # Verify parameters are correctly set
    assert retriever.index_type == 'IndexHNSWFlat', "Index type not set correctly"
    assert retriever.hnsw_m == 32, "HNSW M not set correctly"
    assert retriever.hnsw_ef_construction == 400, "HNSW ef_construction not set correctly"
    assert retriever.hnsw_ef_search == 128, "HNSW ef_search not set correctly"
    
    print("   ✅ HNSW parameters: PASS")
    return True

def test_base_retriever_consistency():
    """Test consistency with BaseRetriever interface."""
    print("\n🏗️ Testing BaseRetriever consistency...")
    
    config_manager = ConfigManager()
    retriever_config = config_manager.get_retriever_config()
    
    retriever = FAISSRetriever(config=retriever_config)
    
    # Test inherited properties
    print(f"   ✅ index_path (inherited): {retriever.index_path}")
    print(f"   ✅ similarity_metric: {retriever.similarity_metric}")
    print(f"   ✅ default_top_k: {retriever.default_top_k}")
    
    # Test methods exist
    assert hasattr(retriever, 'get_chunk_count'), "get_chunk_count method missing"
    assert hasattr(retriever, 'get_statistics'), "get_statistics method missing"
    assert hasattr(retriever, 'clear_index'), "clear_index method missing"
    
    print("   ✅ BaseRetriever consistency: PASS")
    return True

def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE RETRIEVER SYSTEM TEST")
    print("="*70)
    
    tests = [
        test_path_unification,
        test_config_coherence,
        test_hnsw_parameters,
        test_base_retriever_consistency
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ {test_func.__name__} FAILED: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("📋 FINAL RESULTS:")
    
    test_names = [
        "Path Unification",
        "Configuration Coherence", 
        "HNSW Parameters",
        "BaseRetriever Consistency"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    overall_success = all(results)
    print(f"\n🏁 OVERALL SYSTEM: {'✅ EXCELLENT' if overall_success else '❌ NEEDS ATTENTION'}")
    
    return overall_success

if __name__ == "__main__":
    main()