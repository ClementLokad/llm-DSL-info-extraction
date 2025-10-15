#!/usr/bin/env python3
"""
DSL Query System - Comprehensive Test Suite
Professional CLI for automated testing and validation
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json

from config_manager import ConfigManager


@dataclass
class TestResult:
    """Test result container."""
    name: str
    success: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class TestSuite:
    """Test suite container."""
    name: str
    description: str
    results: List[TestResult]
    total_duration: float = 0.0

    def __post_init__(self):
        if not hasattr(self, 'results'):
            self.results = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def passed_count(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if not r.success)


class DSLTestSuite:
    """Comprehensive test suite for DSL Query System."""

    def __init__(self, quiet: bool = False):
        """Initialize test suite."""
        self.config_manager = ConfigManager()
        self.suites: List[TestSuite] = []
        self.quiet = quiet
        
    def run_tests(self, categories: List[str] = None, export_path: str = None) -> Dict[str, Any]:
        """Run specified test categories."""
        if not categories:
            categories = ['config', 'agents', 'pipeline', 'index', 'quality', 'integration']
        
        if not self.quiet:
            print("🧪 DSL QUERY SYSTEM - TEST SUITE")
            print("=" * 50)
            print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📋 Categories: {', '.join(categories)}")
            print("=" * 50)

        start_time = time.time()
        
        # Run selected test categories
        for category in categories:
            if category == 'config':
                self._run_configuration_tests()
            elif category == 'agents':
                self._run_agent_tests()
            elif category == 'pipeline':
                self._run_pipeline_tests()
            elif category == 'index':
                self._run_index_tests()
            elif category == 'quality':
                self._run_quality_tests()
            elif category == 'integration':
                self._run_integration_tests()
            else:
                if not self.quiet:
                    print(f"⚠️ Unknown test category: {category}")

        total_duration = time.time() - start_time
        
        # Generate summary report
        summary = self._generate_summary_report(total_duration)
        
        if not self.quiet:
            self._display_summary_report(summary)
        
        # Export results if requested
        if export_path:
            self._export_results(summary, export_path)
            
        return summary

    def _run_configuration_tests(self) -> None:
        """Test configuration management."""
        suite = TestSuite("Configuration Tests", "Test configuration loading and validation", [])
        
        if not self.quiet:
            print("\n🔧 Configuration Tests")
            print("-" * 30)

        # Test 1: Config file loading
        start_time = time.time()
        try:
            config = self.config_manager.config
            assert isinstance(config, dict), "Config should be a dictionary"
            assert 'agent' in config, "Agent config should be present"
            
            duration = time.time() - start_time
            result = TestResult("Config Loading", True, duration, "Configuration loaded successfully")
            if not self.quiet:
                print(f"✅ Config Loading ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Config Loading", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Config Loading: {e}")
                
        suite.results.append(result)

        # Test 2: Default agent configuration
        start_time = time.time()
        try:
            default_agent = self.config_manager.get_default_agent()
            assert default_agent in ['gemini', 'gpt', 'mistral'], f"Invalid default agent: {default_agent}"
            
            duration = time.time() - start_time
            result = TestResult("Default Agent Config", True, duration, f"Default agent: {default_agent}")
            if not self.quiet:
                print(f"✅ Default Agent: {default_agent} ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Default Agent Config", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Default Agent Config: {e}")
                
        suite.results.append(result)

        # Test 3: API keys presence
        start_time = time.time()
        try:
            default_agent = self.config_manager.get_default_agent()
            key_map = {
                'mistral': 'MISTRAL_API_KEY',
                'gpt': 'OPENAI_API_KEY',
                'gemini': 'GEMINI_API_KEY'
            }
            
            api_key = self.config_manager.get_api_key(key_map[default_agent])
            assert api_key, f"API key missing for {default_agent}"
            
            duration = time.time() - start_time
            result = TestResult("API Key Check", True, duration, f"API key present for {default_agent}")
            if not self.quiet:
                print(f"✅ API Key Check ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("API Key Check", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ API Key Check: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _run_agent_tests(self) -> None:
        """Test AI agent functionality."""
        suite = TestSuite("Agent Tests", "Test AI agent initialization and basic functionality", [])
        
        if not self.quiet:
            print("\n🤖 Agent Tests")
            print("-" * 30)

        # Test agent initialization
        start_time = time.time()
        try:
            default_agent = self.config_manager.get_default_agent()
            
            if default_agent == 'mistral':
                from agents.mistral_agent import MistralAgent
                agent = MistralAgent()
            elif default_agent == 'gpt':
                from agents.gpt_agent import GPTAgent
                agent = GPTAgent()
            elif default_agent == 'gemini':
                from agents.gemini_agent import GeminiAgent
                agent = GeminiAgent()
            else:
                raise ValueError(f"Unknown agent: {default_agent}")
                
            agent.initialize()
            
            duration = time.time() - start_time
            result = TestResult("Agent Initialization", True, duration, f"{agent.model_name} initialized")
            if not self.quiet:
                print(f"✅ Agent Init: {agent.model_name} ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Agent Initialization", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Agent Init: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _run_pipeline_tests(self) -> None:
        """Test pipeline components."""
        suite = TestSuite("Pipeline Tests", "Test pipeline component initialization", [])
        
        if not self.quiet:
            print("\n🔧 Pipeline Tests")
            print("-" * 30)

        # Test pipeline initialization
        start_time = time.time()
        try:
            from pipeline import EnvisionParser, SemanticChunker, SentenceTransformerEmbedder, FAISSRetriever
            
            # Test parser
            parser = EnvisionParser()
            chunker = SemanticChunker()
            embedder = SentenceTransformerEmbedder()
            retriever = FAISSRetriever()
            
            duration = time.time() - start_time
            result = TestResult("Pipeline Components", True, duration, "All components initialized")
            if not self.quiet:
                print(f"✅ Pipeline Components ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Pipeline Components", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Pipeline Components: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _run_index_tests(self) -> None:
        """Test index availability and integrity."""
        suite = TestSuite("Index Tests", "Test knowledge index status", [])
        
        if not self.quiet:
            print("\n📚 Index Tests")
            print("-" * 30)

        # Test index existence
        start_time = time.time()
        try:
            import os
            index_path = "data/faiss_index"
            
            if not os.path.exists(index_path):
                raise FileNotFoundError("Index directory not found")
                
            files = os.listdir(index_path)
            required_files = ['chunks.pkl']  # Minimum required files
            
            for req_file in required_files:
                if req_file not in files:
                    raise FileNotFoundError(f"Required index file missing: {req_file}")
            
            duration = time.time() - start_time
            result = TestResult("Index Availability", True, duration, f"Index found with {len(files)} files")
            if not self.quiet:
                print(f"✅ Index Available: {len(files)} files ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Index Availability", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Index Availability: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _run_quality_tests(self) -> None:
        """Test code quality and standards."""
        suite = TestSuite("Quality Tests", "Test code quality metrics", [])
        
        if not self.quiet:
            print("\n🎯 Quality Tests")
            print("-" * 30)

        # Basic syntax check
        start_time = time.time()
        try:
            import ast
            import os
            
            python_files = []
            for root, dirs, files in os.walk('.'):
                if 'env' in root or '__pycache__' in root:
                    continue
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            syntax_errors = []
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
            
            if syntax_errors:
                raise ValueError(f"Syntax errors found: {len(syntax_errors)}")
            
            duration = time.time() - start_time
            result = TestResult("Syntax Check", True, duration, f"Checked {len(python_files)} Python files")
            if not self.quiet:
                print(f"✅ Syntax Check: {len(python_files)} files ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Syntax Check", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Syntax Check: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _run_integration_tests(self) -> None:
        """Test full system integration."""
        suite = TestSuite("Integration Tests", "Test complete system functionality", [])
        
        if not self.quiet:
            print("\n🔗 Integration Tests")
            print("-" * 30)

        # Test 1: System initialization
        start_time = time.time()
        try:
            # Import main system class
            from main import DSLQuerySystem
            
            # Create system instance
            system = DSLQuerySystem()
            
            # Initialize components (excluding index loading for now)
            system._initialize_pipeline(verbose=False)
            system._initialize_agent(verbose=False)
            
            duration = time.time() - start_time
            result = TestResult("System Components", True, duration, "System components initialized")
            if not self.quiet:
                print(f"✅ System Components ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("System Components", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ System Components: {e}")
                
        suite.results.append(result)
        
        # Test 2: Full system with index loading
        start_time = time.time()
        try:
            # Test complete initialization including index
            system = DSLQuerySystem()
            system.initialize(verbose=False)
            
            duration = time.time() - start_time
            result = TestResult("Full System", True, duration, "Complete system initialized with index")
            if not self.quiet:
                print(f"✅ Full System ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult("Full System", False, duration, f"Failed: {e}")
            if not self.quiet:
                print(f"❌ Full System: {e}")
                
        suite.results.append(result)
        
        suite.total_duration = sum(r.duration for r in suite.results)
        self.suites.append(suite)

    def _generate_summary_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_tests = sum(len(suite.results) for suite in self.suites)
        passed_tests = sum(suite.passed_count for suite in self.suites)
        failed_tests = sum(suite.failed_count for suite in self.suites)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'suites': [
                {
                    'name': suite.name,
                    'description': suite.description,
                    'total_tests': len(suite.results),
                    'passed_tests': suite.passed_count,
                    'failed_tests': suite.failed_count,
                    'success_rate': suite.success_rate,
                    'duration': suite.total_duration,
                    'results': [
                        {
                            'name': result.name,
                            'success': result.success,
                            'duration': result.duration,
                            'message': result.message,
                            'details': result.details
                        }
                        for result in suite.results
                    ]
                }
                for suite in self.suites
            ]
        }

    def _display_summary_report(self, summary: Dict[str, Any]) -> None:
        """Display formatted summary report."""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY REPORT")
        print("=" * 70)
        
        # Overall statistics
        print(f"🕐 Duration: {summary['total_duration']:.2f}s")
        print(f"📋 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        
        # Suite breakdown
        print("\n📝 Test Suite Breakdown:")
        for suite_data in summary['suites']:
            status = "✅" if suite_data['failed_tests'] == 0 else "❌"
            print(f"  {status} {suite_data['name']}: {suite_data['passed_tests']}/{suite_data['total_tests']} ({suite_data['success_rate']:.1%})")
        
        # Failed tests details
        if summary['failed_tests'] > 0:
            print("\n🔍 Failed Tests Details:")
            for suite_data in summary['suites']:
                for result in suite_data['results']:
                    if not result['success']:
                        print(f"  ❌ {suite_data['name']} / {result['name']}: {result['message']}")
        
        print("=" * 70)

    def _export_results(self, summary: Dict[str, Any], export_path: str) -> None:
        """Export test results to JSON file."""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            if not self.quiet:
                print(f"📄 Results exported to: {export_path}")
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ Export failed: {e}")


def main():
    """Main entry point for test suite."""
    parser = argparse.ArgumentParser(
        description="DSL Query System - Comprehensive Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEST CATEGORIES:
  config      : Configuration management tests
  agents      : AI agent functionality tests  
  pipeline    : Pipeline component tests
  index       : Knowledge index tests
  quality     : Code quality and syntax tests
  integration : Full system integration tests

EXAMPLES:
  python test.py                           # Run all tests
  python test.py --categories config      # Run only config tests
  python test.py --categories config agents # Run config and agent tests
  python test.py --quiet                  # Run silently
  python test.py --export results.json    # Export results to JSON
  python test.py --help                   # Show this help
        """
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=["config", "agents", "pipeline", "index", "quality", "integration"],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Run tests silently with minimal output"
    )
    
    parser.add_argument(
        "--export", "-e",
        metavar="FILE",
        help="Export test results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Create test suite
        test_suite = DSLTestSuite(quiet=args.quiet)
        
        # Run tests
        summary = test_suite.run_tests(
            categories=args.categories,
            export_path=args.export
        )
        
        # Exit with appropriate code
        if summary['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n\n👋 Test interrupted!")
        sys.exit(130)
    except Exception as e:
        if not args.quiet:
            print(f"❌ Test suite error: {e}")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()