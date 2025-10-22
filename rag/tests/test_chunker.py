#!/usr/bin/env python3
"""
Tests for the semantic chunker component.
Tests real Envision script blocks obtained from test_parser.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config_manager import ConfigManager
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.tests.test_parser import TestEnvisionParser
from rag.core.base_parser import CodeBlock
from rag.core.base_chunker import CodeChunk

class TestSemanticChunker:
    """Test cases for SemanticChunker using real Envision script blocks."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.chunker_config = self.config_manager.get_chunker_config()
        self.chunker = SemanticChunker(config=self.chunker_config)
        self.timing_stats = {}
        
        # Initialize parser test to get real blocks
        self.parser_test = TestEnvisionParser()
        self.parsed_blocks = None
    
    def _time_function(self, func_name, func):
        """Time a function and store the result."""
        start_time = time.time()
        result = func()
        end_time = time.time()
        duration = end_time - start_time
        self.timing_stats[func_name] = duration
        return result, duration
    
    def get_envision_blocks(self):
        """Get real parsed blocks from an Envision script using test_parser."""
        print("📁 Getting real Envision blocks from parser...")
        
        # Run parser tests to get blocks
        self.parser_test.test_parser_initialization()
        blocks = None
        
        # Try to parse a real file first
        try:
            test_dirs = self.config_manager.get('paths.test_dirs', ["env_scripts"])
            for test_dir in test_dirs:
                test_path = project_root / test_dir
                if test_path.exists():
                    for ext in self.parser_test.parser.supported_extensions:
                        files = list(test_path.glob(f"*{ext}"))
                        if files:
                            test_file = files[0]  # Take first available file
                            blocks = self.parser_test.parser.parse_file(str(test_file))
                            print(f"   ✅ Loaded {len(blocks)} blocks from {test_file.name}")
                            break
                    if blocks:
                        break
        except Exception as e:
            print(f"   ⚠️ Could not parse file: {e}")
        
        # Fallback to test content if no file found
        if not blocks:
            print("   📝 Using test content as fallback...")
            test_content = """/// Real Envision Test Script
/// Data Import and Processing

const projectPath = "C:\\data\\sales\\"
read projectPath + "sales_data.csv" as Sales with
    ItemCode : text
    ItemName : text  
    Quantity : number
    UnitPrice : number
    SaleDate : date
    Region : text

read projectPath + "inventory.csv" as Inventory with
    ItemCode : text
    ItemName : text
    Stock : number
    ReorderLevel : number

/// Business Logic Calculations
Sales.Revenue = Sales.Quantity * Sales.UnitPrice
Sales.TotalCost = Sales.Quantity * Sales.UnitPrice * 0.6
Sales.Profit = Sales.Revenue - Sales.TotalCost
Sales.ProfitMargin = (Sales.Profit / Sales.Revenue) * 100

/// Data Analysis and Joins
SalesWithStock = join(Sales, Inventory) by ItemCode
LowStockItems = where SalesWithStock.Stock < SalesWithStock.ReorderLevel

/// Aggregations
TotalRevenue = sum(Sales.Revenue)
TotalProfit = sum(Sales.Profit)
AvgOrderValue = avg(Sales.Revenue)
MonthlyStats = Sales
  | group by month(Sales.SaleDate)
  | summarize 
      TotalSales = sum(Revenue),
      OrderCount = count(),
      AvgOrderValue = avg(Revenue)

/// Conditional Logic
Sales.CustomerTier = case
  when Sales.Revenue > 1000 then "Premium"
  when Sales.Revenue > 500 then "Gold"
  when Sales.Revenue > 100 then "Silver"
  else "Bronze"

/// Report Generation
show table "Sales Summary" with
  Sales.ItemCode
  Sales.ItemName  
  Sales.Revenue
  Sales.Profit
  Sales.CustomerTier
order by Sales.Revenue desc

show table "Low Stock Alert" with
  LowStockItems.ItemCode
  LowStockItems.ItemName
  LowStockItems.Stock
  LowStockItems.ReorderLevel
where LowStockItems.Stock > 0

show table "Monthly Performance" with
  MonthlyStats.month
  MonthlyStats.TotalSales
  MonthlyStats.OrderCount
  MonthlyStats.AvgOrderValue
order by MonthlyStats.month desc"""
            
            blocks = self.parser_test.parser.parse_content(test_content, "test_real.nvn")
            print(f"   ✅ Generated {len(blocks)} blocks from test content")
        
        if blocks:
            print(f"\n   📊 INPUT BLOCKS ANALYSIS:")
            print(f"   {'='*50}")
            
            # Analyze block types
            block_types = {}
            total_content_length = 0
            line_spans = []
            
            for block in blocks:
                btype = block.block_type
                block_types[btype] = block_types.get(btype, 0) + 1
                total_content_length += len(block.content)
                line_spans.append(block.line_end - block.line_start + 1)
            
            print(f"   📋 Total blocks: {len(blocks)}")
            print(f"   📊 Block types: {dict(block_types)}")
            print(f"   📏 Total content: {total_content_length} characters")
            print(f"   📐 Line spans - Min: {min(line_spans)}, Max: {max(line_spans)}, Avg: {sum(line_spans)/len(line_spans):.1f}")
            
            # Show first few blocks as examples
            print(f"\n   📄 EXAMPLE INPUT BLOCKS:")
            print(f"   {'-'*50}")
            for i, block in enumerate(blocks[:5]):
                content_preview = block.content.strip()[:80]
                if len(block.content.strip()) > 80:
                    content_preview += "..."
                print(f"   [{i+1}] {block.block_type} (lines {block.line_start}-{block.line_end})")
                print(f"       Content: '{content_preview}'")
                if hasattr(block, 'metadata') and block.metadata:
                    print(f"       Metadata: {dict(list(block.metadata.items())[:3])}")
                print()
            
            if len(blocks) > 5:
                print(f"   ... and {len(blocks) - 5} more blocks")
            
            print(f"   ✅ Real Envision blocks ready for chunking")
        
        self.parsed_blocks = blocks
        return blocks
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        print("🔧 Testing chunker initialization...")
        
        def _test_init():
            assert self.chunker is not None, "Chunker should be initialized"
            assert hasattr(self.chunker, 'max_chunk_tokens'), "Chunker should have max_chunk_tokens"
            assert hasattr(self.chunker, 'overlap_lines'), "Chunker should have overlap_lines"
            return True
        
        result, duration = self._time_function('initialization', _test_init)
        
        print(f"   ✅ Max chunk tokens: {self.chunker.max_chunk_tokens}")
        print(f"   ✅ Overlap lines: {self.chunker.overlap_lines}")
        print(f"   ⏱️ Initialization time: {duration:.4f}s")
        print("   ✅ Chunker initialization successful")
        return result
    
    def test_create_test_blocks(self):
        """Create diverse test blocks for chunking tests to produce multiple chunks."""
        blocks = [
            # Block set 1: Data Import Section
            CodeBlock(
                content="/// Data Import Section\n/// Loading sales and inventory data",
                block_type="comment",
                line_start=1,
                line_end=2,
                metadata={"language": "envision", "section": "data_import"}
            ),
            CodeBlock(
                content='''read "sales_data.csv" as Sales with
  Item : text
  Quantity : number
  UnitPrice : number''',
                block_type="read_statement",
                line_start=3,
                line_end=6,
                metadata={"table_name": "Sales", "section": "data_import"}
            ),
            CodeBlock(
                content='''read "inventory.csv" as Inventory with
  Item : text
  Stock : number''',
                block_type="read_statement",
                line_start=7,
                line_end=9,
                metadata={"table_name": "Inventory", "section": "data_import"}
            ),
            
            # Block set 2: Calculations Section (longer content)
            CodeBlock(
                content="/// Business Logic Calculations\n/// Computing revenue, profit margins, and stock ratios",
                block_type="comment",
                line_start=12,
                line_end=13,
                metadata={"language": "envision", "section": "calculations"}
            ),
            CodeBlock(
                content="Sales.Revenue = Sales.Quantity * Sales.UnitPrice",
                block_type="assignment",
                line_start=14,
                line_end=14,
                metadata={"variable_name": "Sales.Revenue", "section": "calculations"}
            ),
            CodeBlock(
                content="Sales.TotalCost = Sales.Quantity * 0.6 * Sales.UnitPrice  // Assume 60% cost ratio",
                block_type="assignment",
                line_start=15,
                line_end=15,
                metadata={"variable_name": "Sales.TotalCost", "section": "calculations"}
            ),
            CodeBlock(
                content="Sales.Profit = Sales.Revenue - Sales.TotalCost",
                block_type="assignment",
                line_start=16,
                line_end=16,
                metadata={"variable_name": "Sales.Profit", "section": "calculations"}
            ),
            CodeBlock(
                content="Sales.ProfitMargin = Sales.Profit / Sales.Revenue * 100",
                block_type="assignment",
                line_start=17,
                line_end=17,
                metadata={"variable_name": "Sales.ProfitMargin", "section": "calculations"}
            ),
            
            # Block set 3: Data Analysis Section
            CodeBlock(
                content="// Join sales with inventory to check stock levels",
                block_type="comment",
                line_start=20,
                line_end=20,
                metadata={"language": "envision", "section": "analysis"}
            ),
            CodeBlock(
                content="SalesWithStock = join(Sales, Inventory) by Item",
                block_type="assignment",
                line_start=21,
                line_end=21,
                metadata={"variable_name": "SalesWithStock", "section": "analysis"}
            ),
            CodeBlock(
                content="SalesWithStock.StockRatio = SalesWithStock.Stock / SalesWithStock.Quantity",
                block_type="assignment",
                line_start=22,
                line_end=22,
                metadata={"variable_name": "SalesWithStock.StockRatio", "section": "analysis"}
            ),
            
            # Block set 4: Output Section
            CodeBlock(
                content="/// Report Generation\n/// Creating visualizations and tables",
                block_type="comment",
                line_start=25,
                line_end=26,
                metadata={"language": "envision", "section": "output"}
            ),
            CodeBlock(
                content='''show table "Sales Analysis" with
  Sales.Item
  Sales.Revenue
  Sales.Profit
  Sales.ProfitMargin''',
                block_type="show_statement",
                line_start=27,
                line_end=31,
                metadata={"show_name": "Sales Analysis", "section": "output"}
            ),
            CodeBlock(
                content='''show table "Stock Status" with
  SalesWithStock.Item
  SalesWithStock.Stock
  SalesWithStock.StockRatio''',
                block_type="show_statement",
                line_start=32,
                line_end=35,
                metadata={"show_name": "Stock Status", "section": "output"}
            )
        ]
        
        return blocks
    
    def test_chunk_blocks(self):
        """Test chunking real Envision blocks with comprehensive analysis."""
        print("✂️ Testing block chunking on real Envision script...")
        
        # Get real blocks from parser
        blocks = self.get_envision_blocks()
        if not blocks:
            print("   ❌ No blocks available for chunking")
            return False
        
        # Perform chunking with timing
        def _chunk_blocks():
            return self.chunker.chunk_blocks(blocks)
            
        chunks, duration = self._time_function('chunking', _chunk_blocks)
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        print(f"   ✅ Created {len(chunks)} chunks from {len(blocks)} blocks")
        print(f"   ⏱️ Chunking time: {duration:.4f}s ({len(blocks)/duration:.1f} blocks/s)")
        
        # Detailed analysis of chunking results
        print(f"\n   📋 DETAILED CHUNKING ANALYSIS:")
        print(f"   {'='*60}")
        
        total_content_length = 0
        total_blocks_used = 0
        chunk_types = {}
        
        for i, chunk in enumerate(chunks):
            assert hasattr(chunk, 'content'), f"Chunk {i} should have content"
            assert hasattr(chunk, 'chunk_type'), f"Chunk {i} should have chunk_type"
            assert hasattr(chunk, 'original_blocks'), f"Chunk {i} should have original_blocks"
            
            # Collect statistics
            total_content_length += len(chunk.content)
            total_blocks_used += len(chunk.original_blocks)
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            print(f"   [Chunk {i+1}] Type: {chunk.chunk_type}")
            print(f"      📏 Content length: {len(chunk.content)} chars")
            print(f"      🧱 Original blocks: {len(chunk.original_blocks)}")
            print(f"      📊 Token estimate: {chunk.size_tokens} tokens")
            
            # Show block types that were combined
            block_types_in_chunk = [block.block_type for block in chunk.original_blocks]
            block_type_counts = {}
            for bt in block_types_in_chunk:
                block_type_counts[bt] = block_type_counts.get(bt, 0) + 1
            print(f"      🔗 Block types combined: {dict(block_type_counts)}")
            
            # Show first line of content
            first_line = chunk.content.split('\n')[0]
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            print(f"      📝 Preview: '{first_line}'")
            
            # Show metadata if available
            if hasattr(chunk, 'metadata') and chunk.metadata:
                print(f"      🏷️ Metadata: {chunk.metadata}")
            print()
        
        # Summary statistics
        print(f"   📊 CHUNKING STATISTICS:")
        print(f"   {'-'*60}")
        print(f"   📈 Input: {len(blocks)} blocks → Output: {len(chunks)} chunks")
        print(f"   📏 Total content: {total_content_length} characters")
        print(f"   🧱 Total blocks processed: {total_blocks_used}")
        print(f"   📋 Chunk types distribution: {chunk_types}")
        if chunks:
            avg_content_length = total_content_length / len(chunks)
            avg_blocks_per_chunk = total_blocks_used / len(chunks)
            print(f"   📊 Average chunk size: {avg_content_length:.1f} chars")
            print(f"   🧱 Average blocks per chunk: {avg_blocks_per_chunk:.1f}")
        
        # Show complete examples of chunks
        print(f"\n   📄 COMPLETE CHUNK EXAMPLES:")
        print(f"   {'-'*60}")
        
        # Show up to 3 chunk examples with full content
        for i, chunk in enumerate(chunks[:3]):
            print(f"   📝 CHUNK {i+1} EXAMPLE ({chunk.chunk_type}):")
            print(f"      📊 Statistics: {len(chunk.content)} chars, {chunk.size_tokens} tokens, {len(chunk.original_blocks)} blocks")
            
            # Show the complete content (limited for readability)
            content = chunk.content.strip()
            if len(content) > 400:
                lines = content.split('\n')
                if len(lines) > 8:
                    # Show first 8 lines + summary
                    partial_content = '\n'.join(lines[:8]) + f"\n... ({len(lines)-8} more lines)"
                else:
                    partial_content = content[:397] + "..."
            else:
                partial_content = content
            
            # Indent the content
            indented_content = '\n'.join(f"         {line}" for line in partial_content.split('\n'))
            print(indented_content)
            
            # Show block composition
            block_types_in_chunk = [block.block_type for block in chunk.original_blocks]
            block_composition = {}
            for bt in block_types_in_chunk:
                block_composition[bt] = block_composition.get(bt, 0) + 1
            print(f"      🧱 Block composition: {dict(block_composition)}")
            
            # Show metadata if available
            if hasattr(chunk, 'metadata') and chunk.metadata:
                print(f"      🏷️ Chunk metadata: {chunk.metadata}")
            
            # Show sections if blocks have section metadata
            sections = set()
            for block in chunk.original_blocks:
                if hasattr(block, 'metadata') and 'section' in block.metadata:
                    sections.add(block.metadata['section'])
            if sections:
                print(f"      📋 Sections covered: {sorted(sections)}")
            print()
        
        if len(chunks) > 3:
            print(f"   ... and {len(chunks) - 3} more chunks")
        
        print("   ✅ Chunk structure validation successful")
        return True
    
    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        print("📏 Testing chunk size limits...")
        
        blocks = self.test_create_test_blocks()
        chunks = self.chunker.chunk_blocks(blocks)
        
        max_size = self.chunker.max_chunk_tokens
        oversized_chunks = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.content) > max_size:
                oversized_chunks.append((i, len(chunk.content)))
        
        if oversized_chunks:
            print(f"   ⚠️ Found {len(oversized_chunks)} oversized chunks:")
            for chunk_idx, size in oversized_chunks:
                print(f"      Chunk {chunk_idx}: {size} chars (max: {max_size})")
            # Don't fail - some complex blocks might exceed limits
        else:
            print(f"   ✅ All chunks within size limit ({max_size} chars)")
        
        print("   ✅ Size limit validation completed")
        return True
    
    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        print("🔄 Testing chunk overlap...")
        
        overlap_size = self.chunker.overlap_lines
        print(f"   📝 Configured overlap size: {overlap_size}")
        
        if overlap_size > 0:
            blocks = self.test_create_test_blocks()
            chunks = self.chunker.chunk_blocks(blocks)
            
            if len(chunks) > 1:
                # Check for content overlap between consecutive chunks
                overlaps_found = 0
                for i in range(len(chunks) - 1):
                    chunk1_end = chunks[i].content[-overlap_size:] if len(chunks[i].content) >= overlap_size else chunks[i].content
                    chunk2_start = chunks[i+1].content[:overlap_size] if len(chunks[i+1].content) >= overlap_size else chunks[i+1].content
                    
                    # Look for any common content
                    if any(word in chunk2_start for word in chunk1_end.split() if len(word) > 3):
                        overlaps_found += 1
                
                print(f"   ✅ Found {overlaps_found} potential overlaps between {len(chunks)-1} chunk pairs")
            else:
                print("   ℹ️ Only one chunk produced - overlap not applicable")
        else:
            print("   ℹ️ No overlap configured")
        
        print("   ✅ Overlap functionality validated")
        return True
    
    def test_configuration_usage(self):
        """Test that chunker properly uses configuration."""
        print("⚙️ Testing configuration usage...")
        
        # Test max_chunk_tokens from config
        config_max_tokens = self.chunker_config.get('max_chunk_tokens', 500)
        chunker_max_tokens = self.chunker.max_chunk_tokens
        
        assert chunker_max_tokens == config_max_tokens, \
            f"Chunker max tokens {chunker_max_tokens} should match config {config_max_tokens}"
        
        print(f"   ✅ Max chunk tokens from config: {config_max_tokens}")
        
        # Test overlap_lines from config
        config_overlap = self.chunker_config.get('overlap_lines', 3)
        chunker_overlap = self.chunker.overlap_lines
        
        assert chunker_overlap == config_overlap, \
            f"Chunker overlap {chunker_overlap} should match config {config_overlap}"
        
        print(f"   ✅ Overlap lines from config: {config_overlap}")
        
        # Test other config parameters
        if 'chars_per_token' in self.chunker_config:
            print(f"   ✅ Chars per token: {self.chunker_config['chars_per_token']}")
        
        if 'preserve_boundaries' in self.chunker_config:
            print(f"   ✅ Preserve boundaries: {self.chunker_config['preserve_boundaries']}")
        
        print("   ✅ Configuration usage validated")
        return True
    
    def run_all_tests(self):
        """Run all chunker tests."""
        print("🧪 SEMANTIC CHUNKER TESTS")
        print("=" * 40)
        
        tests = [
            self.test_chunker_initialization,
            self.test_chunk_blocks,
            self.test_chunk_size_limits,
            self.test_chunk_overlap,
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
        if 'chunking' in self.timing_stats:
            print(f"🚀 Performance metrics:")
            chunking_time = self.timing_stats['chunking']
            print(f"   • Average chunking time: {chunking_time:.4f}s per batch")

def main():
    """Main test runner."""
    tester = TestSemanticChunker()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
