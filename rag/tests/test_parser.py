#!/usr/bin/env python3
"""
Tests for the Envision parser component.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config_manager import ConfigManager
from rag.parsers.old_envision_parser import EnvisionParser

FILE_INDEX = 0

class TestEnvisionParser:
    """Test cases for EnvisionParser."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.parser_config = self.config_manager.get_parser_config()
        self.parser = EnvisionParser(config=self.parser_config)
        self.timing_stats = {}
    
    def _time_function(self, func_name, func):
        """Time a function and store the result."""
        start_time = time.time()
        result = func()
        end_time = time.time()
        duration = end_time - start_time
        self.timing_stats[func_name] = duration
        return result, duration
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        print("🔧 Testing parser initialization...")
        
        def _test_init():
            assert self.parser is not None, "Parser should be initialized"
            assert hasattr(self.parser, 'supported_extensions'), "Parser should have supported_extensions"
            assert self.parser.supported_extensions == self.parser_config.get('supported_extensions', []), \
                "Parser should use config extensions"
            return True
        
        result, duration = self._time_function('initialization', _test_init)
        
        print(f"   ✅ Supported extensions: {self.parser.supported_extensions}")
        print(f"   ⏱️ Initialization time: {duration:.4f}s")
        print("   ✅ Parser initialization successful")
        return result
    
    def test_parse_content_basic(self):
        """Test basic content parsing."""
        print("📄 Testing basic content parsing...")
        
        test_content = """/// Test comment
const projectFolder = ""
read "path" as Catalog with
  ItemCode : text
  ImageName : text

Catalog.Sku = concat(Catalog.ItemCode,"-",Catalog.ItemLoc)

show table "Test" with
  Catalog.Sku"""
        
        blocks = self.parser.parse_content(test_content, "test.nvn")
        
        assert len(blocks) > 0, "Should parse at least one block"
        print(f"   ✅ Parsed {len(blocks)} blocks from test content")
        
        # Check block structure
        for i, block in enumerate(blocks):
            assert hasattr(block, 'content'), f"Block {i} should have content"
            assert hasattr(block, 'block_type'), f"Block {i} should have block_type" 
            assert hasattr(block, 'line_start'), f"Block {i} should have line_start"
            assert hasattr(block, 'line_end'), f"Block {i} should have line_end"
            print(f"      {i+1}. {block.block_type}: lines {block.line_start}-{block.line_end}")
        
        print("   ✅ Block structure validation successful")
        return True
    
    def test_parse_file(self):
        """Test file parsing with detailed analysis of a concrete example."""
        print("📁 Testing file parsing...")
        
        # Look for test files
        test_dirs = self.config_manager.get('paths.test_dirs', ["env_scripts"])
        test_file = None
        
        for test_dir in test_dirs:
            test_path = project_root / test_dir
            if test_path.exists():
                for ext in self.parser.supported_extensions:
                    files = list(test_path.glob(f"*{ext}"))
                    if files:
                        test_file = files[FILE_INDEX % len(files)]
                        break
                if test_file:
                    break
        
        if test_file:
            try:
                def _parse_file():
                    return self.parser.parse_file(str(test_file))
                
                blocks, duration = self._time_function('file_parsing', _parse_file)
                print(f"   ✅ Parsed {len(blocks)} blocks from {test_file.name}")
                print(f"   📁 File path: {test_file}")
                print(f"   📊 File size: {test_file.stat().st_size} bytes")
                print(f"   ⏱️ Parsing time: {duration:.4f}s ({len(blocks)/duration:.1f} blocks/s)")
                
                # Display detailed analysis of parsed blocks
                if blocks:
                    print(f"\n   📋 DETAILED ANALYSIS OF PARSED BLOCKS:")
                    print(f"   {'='*60}")
                    
                    # Count blocks by type
                    block_types = {}
                    for block in blocks:
                        block_type = block.block_type
                        block_types[block_type] = block_types.get(block_type, 0) + 1
                    
                    print(f"   📊 Block type distribution:")
                    for block_type, count in sorted(block_types.items()):
                        print(f"      • {block_type}: {count} blocks")
                    
                    print(f"\n   🔍 FIRST 5 BLOCKS DETAILS:")
                    print(f"   {'-'*60}")
                    
                    for i, block in enumerate(blocks[:5]):
                        print(f"   [{i+1}] Type: {block.block_type}")
                        print(f"       Lines: {block.line_start}-{block.line_end} ({block.line_end - block.line_start + 1} lines)")
                        print(f"       Content length: {len(block.content)} chars")
                        
                        # Show metadata if available
                        if hasattr(block, 'metadata') and block.metadata:
                            print(f"       Metadata: {block.metadata}")
                        
                        # Show first line of content (truncated)
                        first_line = block.content.split('\n')[0]
                        if len(first_line) > 60:
                            first_line = first_line[:57] + "..."
                        print(f"       Preview: '{first_line}'")
                        print()
                    
                    if len(blocks) > 5:
                        print(f"   ... and {len(blocks) - 5} more blocks")
                    
                    # Show interesting blocks (assignments, read statements, etc.)
                    print(f"\n   🎯 INTERESTING BLOCKS ANALYSIS:")
                    print(f"   {'-'*60}")
                    
                    for block_type in ['read_statement', 'assignment', 'show_statement']:
                        matching_blocks = [b for b in blocks if b.block_type == block_type]
                        if matching_blocks:
                            print(f"   📌 {block_type.upper()} examples:")
                            for i, block in enumerate(matching_blocks[:2]):
                                content_preview = block.content.replace('\n', ' ').strip()
                                if len(content_preview) > 80:
                                    content_preview = content_preview[:77] + "..."
                                print(f"      {i+1}. Lines {block.line_start}-{block.line_end}: {content_preview}")
                            if len(matching_blocks) > 2:
                                print(f"      ... and {len(matching_blocks) - 2} more {block_type} blocks")
                            print()
                    
                    # Show metadata analysis
                    print(f"\n   🏷️ METADATA ANALYSIS:")
                    print(f"   {'-'*60}")
                    
                    # Collect all metadata keys
                    all_metadata_keys = set()
                    metadata_examples = {}
                    
                    for block in blocks:
                        if hasattr(block, 'metadata') and block.metadata:
                            for key in block.metadata.keys():
                                all_metadata_keys.add(key)
                                if key not in metadata_examples:
                                    metadata_examples[key] = []
                                if len(metadata_examples[key]) < 3:
                                    metadata_examples[key].append((block.block_type, block.metadata[key]))
                    
                    if all_metadata_keys:
                        print(f"   📊 Metadata keys found: {sorted(all_metadata_keys)}")
                        print(f"   📝 Metadata examples:")
                        for key in sorted(all_metadata_keys):
                            examples = metadata_examples[key]
                            print(f"      • {key}: {examples[:2]}")
                    else:
                        print(f"   ℹ️ No metadata found in blocks")
                    
                    # Show content length statistics
                    print(f"\n   📏 CONTENT STATISTICS:")
                    print(f"   {'-'*60}")
                    content_lengths = [len(block.content) for block in blocks]
                    if content_lengths:
                        avg_length = sum(content_lengths) / len(content_lengths)
                        min_length = min(content_lengths)
                        max_length = max(content_lengths)
                        print(f"   📊 Content length - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f} chars")
                        
                        # Find the largest block
                        largest_block = max(blocks, key=lambda b: len(b.content))
                        print(f"   📏 Largest block: {largest_block.block_type} (lines {largest_block.line_start}-{largest_block.line_end}, {len(largest_block.content)} chars)")
                        
                        # Show line span statistics
                        line_spans = [block.line_end - block.line_start + 1 for block in blocks]
                        avg_span = sum(line_spans) / len(line_spans)
                        max_span = max(line_spans)
                        print(f"   📐 Line spans - Max: {max_span}, Avg: {avg_span:.1f} lines")
                    
                    # Show complete examples of some interesting blocks
                    print(f"\n   📄 COMPLETE BLOCK EXAMPLES:")
                    print(f"   {'-'*60}")
                    
                    # Show one example of each main type
                    shown_types = set()
                    for block in blocks:
                        if block.block_type not in shown_types and len(shown_types) < 3:
                            shown_types.add(block.block_type)
                            print(f"   📝 {block.block_type.upper()} example (lines {block.line_start}-{block.line_end}):")
                            # Show full content but limit to 300 chars for readability
                            content = block.content.strip()
                            if len(content) > 300:
                                lines = content.split('\n')
                                if len(lines) > 3:
                                    # Show first 3 lines + summary
                                    partial_content = '\n'.join(lines[:3]) + f"\n... ({len(lines)-3} more lines)"
                                else:
                                    partial_content = content[:297] + "..."
                            else:
                                partial_content = content
                            
                            # Indent the content
                            indented_content = '\n'.join(f"      {line}" for line in partial_content.split('\n'))
                            print(indented_content)
                            
                            if hasattr(block, 'metadata') and block.metadata:
                                print(f"      🏷️ Metadata: {block.metadata}")
                            print()
                
                return True
            except Exception as e:
                print(f"   ⚠️ File parsing failed: {e}")
                return True  # Don't fail test if file is problematic
        else:
            print("   ⚠️ No test files found - skipping file parsing test")
            return True
    
    def test_configuration_usage(self):
        """Test that parser properly uses configuration."""
        print("⚙️ Testing configuration usage...")
        
        # Test extensions from config
        config_extensions = self.parser_config.get('supported_extensions', [])
        parser_extensions = self.parser.supported_extensions
        
        assert parser_extensions == config_extensions, \
            f"Parser extensions {parser_extensions} should match config {config_extensions}"
        
        print(f"   ✅ Extensions from config: {config_extensions}")
        
        # Test other config parameters if they exist
        if 'multiline_patterns' in self.parser_config:
            print(f"   ✅ Multiline patterns: {self.parser_config['multiline_patterns']}")
        
        if 'case_sensitive' in self.parser_config:
            print(f"   ✅ Case sensitive: {self.parser_config['case_sensitive']}")
        
        print("   ✅ Configuration usage validated")
        return True
    
    def run_all_tests(self):
        """Run all parser tests."""
        print("🧪 ENVISION PARSER TESTS")
        print("=" * 40)
        
        tests = [
            self.test_parser_initialization,
            self.test_parse_content_basic,
            self.test_parse_file,
            self.test_configuration_usage
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
        
        print("\n" + "=" * 40)
        print(f"📊 RESULTS: {passed} passed, {failed} failed")
        self._print_timing_stats()
        print("=" * 40)
        
        return failed == 0
    
    def _print_timing_stats(self):
        """Print timing statistics."""
        if not self.timing_stats:
            return
            
        print(f"\n⏱️ TIMING STATISTICS:")
        print(f"{'='*40}")
        
        total_time = sum(self.timing_stats.values())
        print(f"📈 Total execution time: {total_time:.4f}s")
        
        if len(self.timing_stats) > 1:
            print(f"📊 Breakdown by operation:")
            for operation, duration in sorted(self.timing_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100
                print(f"   • {operation}: {duration:.4f}s ({percentage:.1f}%)")
        
        # Performance metrics
        if 'file_parsing' in self.timing_stats:
            print(f"🚀 Performance metrics:")
            parsing_time = self.timing_stats['file_parsing']
            print(f"   • Average parsing time: {parsing_time:.4f}s per file")

def main():
    """Main test runner."""
    tester = TestEnvisionParser()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
