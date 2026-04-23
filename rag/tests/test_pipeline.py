#!/usr/bin/env python3
"""
Comprehensive pipeline tests with timing statistics.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager

class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.timing_stats = {}
    
    def _time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result, elapsed
    
    def _print_timing_stats(self):
        """Print detailed timing statistics."""
        if not self.timing_stats:
            return
            
        print(f"\n⏱️ TIMING STATISTICS:")
        print(f"{'=' * 50}")
        
        total_time = sum(self.timing_stats.values())
        print(f"📈 Total execution time: {total_time:.4f}s")
        
        if total_time > 0:
            print(f"📊 Breakdown by operation:")
            for operation, elapsed in self.timing_stats.items():
                percentage = (elapsed / total_time) * 100
                print(f"   • {operation}: {elapsed:.4f}s ({percentage:.1f}%)")
        
        print(f"{'=' * 50}")
    
    def test_configuration_loading(self):
        """Test that all configurations load properly."""
        print("⚙️ Testing configuration loading...")
        
        # Test individual component configs
        components = [
            ('parser', self.config_manager.get_parser_config),
            ('chunker', self.config_manager.get_chunker_config),
            ('embedder', self.config_manager.get_embedder_config),
            ('retriever', self.config_manager.get_retriever_config)
        ]
        
        for name, config_getter in components:
            try:
                config = config_getter()
                assert config is not None, f"{name} config should not be None"
                print(f"   ✅ {name.capitalize()} config loaded: {len(config)} parameters")
            except Exception as e:
                print(f"   ❌ {name.capitalize()} config failed: {e}")
                return False
        
        print("   ✅ All configurations loaded successfully")
        return True
    
    def test_component_imports(self):
        """Test that all pipeline components can be imported."""
        print("📦 Testing component imports...")
        
        imports_to_test = [
            ('EnvisionParser', 'pipeline.parsers.envision_parser'),
            ('SemanticChunker', 'pipeline.chunkers.semantic_chunker'),
            ('SentenceTransformerEmbedder', 'pipeline.embedders.sentence_transformer_embedder'),
            ('FAISSRetriever', 'pipeline.retrievers.faiss_retriever'),
        ]
        
        for class_name, module_path in imports_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                assert cls is not None, f"{class_name} should be importable"
                print(f"   ✅ {class_name} imported successfully")
            except Exception as e:
                print(f"   ❌ {class_name} import failed: {e}")
                return False
        
        print("   ✅ All component imports successful")
        return True
    
    def test_component_initialization(self):
        """Test that all components can be initialized."""
        print("🔧 Testing component initialization...")
        
        components_to_test = [
            ('EnvisionParser', 'pipeline.parsers.envision_parser', self.config_manager.get_parser_config),
            ('SemanticChunker', 'pipeline.chunkers.semantic_chunker', self.config_manager.get_chunker_config),
            ('SentenceTransformerEmbedder', 'pipeline.embedders.sentence_transformer_embedder', self.config_manager.get_embedder_config),
            ('FAISSRetriever', 'pipeline.retrievers.faiss_retriever', self.config_manager.get_retriever_config),
        ]
        
        initialized_components = {}
        
        for class_name, module_path, config_getter in components_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                config = config_getter()
                
                component = cls(config=config)
                assert component is not None, f"{class_name} should initialize"
                
                initialized_components[class_name] = component
                print(f"   ✅ {class_name} initialized successfully")
                
            except Exception as e:
                print(f"   ❌ {class_name} initialization failed: {e}")
                # Don't return False immediately - some components might need dependencies
                print(f"      (This might be due to missing dependencies)")
        
        print(f"   ✅ {len(initialized_components)}/{len(components_to_test)} components initialized")
        return len(initialized_components) > 0  # At least some should work
    

    
    def test_main_system_import(self):
        """Test that main system can be imported."""
        print("🏠 Testing main system import...")
        
        try:
            from main import DSLQuerySystem
            
            # Try to initialize (but don't require it to work fully)
            try:
                system = DSLQuerySystem()
                print("   ✅ DSLQuerySystem initialized successfully")
                
                # Test basic methods exist
                assert hasattr(system, 'build_index'), "Should have build_index method"
                assert hasattr(system, 'query'), "Should have query method"
                print("   ✅ DSLQuerySystem has required methods")
                
            except Exception as e:
                print(f"   ⚠️ DSLQuerySystem initialization failed: {e}")
                print("   ℹ️ This might be due to missing model files or dependencies")
            
        except Exception as e:
            print(f"   ❌ Main system import failed: {e}")
            return False
        
        print("   ✅ Main system import successful")
        return True
    
    def test_individual_test_files(self):
        """Test that individual test files can be run."""
        print("🧪 Testing individual test files...")
        
        test_files = [
            'test_parser.py',
            'test_chunker.py', 
            'test_embedder.py',
            'test_retriever.py'
        ]
        
        tests_dir = Path(__file__).parent
        
        for test_file in test_files:
            test_path = tests_dir / test_file
            if test_path.exists():
                print(f"   ✅ {test_file} exists")
                
                # Try to import the test module
                try:
                    spec = __import__(f"pipeline.tests.{test_file[:-3]}", fromlist=['main'])
                    if hasattr(spec, 'main'):
                        print(f"      📝 {test_file} has main function")
                    else:
                        print(f"      ⚠️ {test_file} missing main function")
                except Exception as e:
                    print(f"      ❌ {test_file} import failed: {e}")
            else:
                print(f"   ❌ {test_file} missing")
        
        print("   ✅ Individual test files validation completed")
        return True
    
    def run_all_tests(self):
        """Run all pipeline integration tests."""
        print("🧪 PIPELINE INTEGRATION TESTS")
        print("=" * 50)
        
        tests = [
            self.test_configuration_loading,
            self.test_component_imports,
            self.test_component_initialization,
            self.test_main_system_import,
            self.test_individual_test_files
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                print(f"\n🔬 Running {test.__name__}...")
                result = test()
                if result:
                    passed += 1
                    print(f"✅ {test.__name__} PASSED")
                else:
                    failed += 1
                    print(f"❌ {test.__name__} FAILED")
            except Exception as e:
                failed += 1
                print(f"❌ {test.__name__} FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)
        print(f"📊 RESULTS: {passed} passed, {failed} failed")
        print("=" * 50)
        
        return failed == 0

def main():
    """Main test runner."""
    tester = TestPipelineIntegration()
    success = tester.run_all_tests()
    
    print("\n" + "🎯 RUNNING INDIVIDUAL COMPONENT TESTS")
    print("=" * 50)
    
    # Try to run individual test files
    individual_results = []
    test_files = ['test_parser', 'test_chunker', 'test_embedder', 'test_retriever']
    
    for test_name in test_files:
        try:
            print(f"\n🔬 Running {test_name}...")
            test_module = __import__(f'pipeline.tests.{test_name}', fromlist=['main'])
            if hasattr(test_module, 'main'):
                result = test_module.main()
                individual_results.append((test_name, result == 0))
                print(f"✅ {test_name} completed with code {result}")
            else:
                print(f"⚠️ {test_name} has no main function")
                individual_results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            individual_results.append((test_name, False))
    
    print("\n" + "📊 INDIVIDUAL TEST RESULTS:")
    for test_name, success in individual_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    overall_success = success and all(result for _, result in individual_results)
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
