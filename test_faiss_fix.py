#!/usr/bin/env python3
"""Test script to validate FAISS retriever fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from preprocessing.retrievers.faiss_retriever import FAISSRetriever

def test_faiss_initialization():
    """Test that FAISSRetriever initializes correctly with index_path"""
    print("🧪 Testing FAISS Retriever initialization...")
    
    # Initialize config
    config_manager = ConfigManager()
    retriever_config = config_manager.get_retriever_config()
    # Pass the full retriever config which includes both general settings and faiss subsection
    
    print(f"📋 Config loaded: {retriever_config}")
    
    # Test FAISSRetriever initialization
    try:
        retriever = FAISSRetriever(config=retriever_config)
        
        # Check that index_path is properly defined
        print(f"✅ index_path correctly defined: {retriever.index_path}")
        
        # Verify the paths can be constructed (even though not stored as attributes)
        expected_chunks_file = retriever._get_chunks_file_path()
        expected_index_file = retriever._get_index_file_path()
        print(f"✅ chunks_file path would be: {expected_chunks_file}")
        print(f"✅ index_file path would be: {expected_index_file}")
        
        # Verify configuration
        print(f"✅ index_type: {retriever.index_type}")
        print(f"✅ use_gpu: {retriever.use_gpu}")
        print(f"✅ nlist: {retriever.nlist}")
        print(f"✅ nprobe: {retriever.nprobe}")
        
        print("🎉 FAISS Retriever initialization: SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

if __name__ == "__main__":
    success = test_faiss_initialization()
    print(f"\n{'='*50}")
    print(f"🏁 Final Result: {'PASS' if success else 'FAIL'}")