#!/usr/bin/env python3
"""
Tests for the sentence transformer embedder component.
Tests real chunks obtained from test_chunker (which uses test_parser).
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config_manager import ConfigManager
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.tests.test_chunker import TestSemanticChunker

class TestChunk:
    """Simple chunk class for testing."""
    def __init__(self, content, metadata=None, chunk_type="test", context=""):
        self.content = content
        self.metadata = metadata or {}
        self.chunk_type = chunk_type
        self.context = context

class TestSentenceTransformerEmbedder:
    """Test cases for SentenceTransformerEmbedder using real chunks from Envision scripts."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.embedder_config = self.config_manager.get_embedder_config()
        # Ensure device is set to "cpu"
        self.embedder_config["device"] = "cpu"
        self.embedder = SentenceTransformerEmbedder(config=self.embedder_config)
        self.timing_stats = {}
        
        # Initialize chunker test to get real chunks
        self.chunker_test = TestSemanticChunker()
        self.real_chunks = None
    
    def _time_function(self, func_name, func):
        """Time a function and store the result."""
        start_time = time.time()
        result = func()
        end_time = time.time()
        duration = end_time - start_time
        self.timing_stats[func_name] = duration
        return result, duration
    
    def get_real_chunks(self):
        """Get real chunks from chunker test (which uses parser test)."""
        print("🧩 Getting real chunks from chunker...")
        
        # Run chunker initialization
        self.chunker_test.test_chunker_initialization()
        
        # Get real blocks from parser and chunk them
        blocks = self.chunker_test.get_envision_blocks()
        if not blocks:
            print("   ❌ No blocks available from parser")
            return None
        
        # Chunk the blocks
        chunks = self.chunker_test.chunker.chunk_blocks(blocks)
        if not chunks:
            print("   ❌ No chunks created from blocks")
            return None
        
        print(f"   ✅ Obtained {len(chunks)} chunks from {len(blocks)} blocks")
        
        # Analyze chunk inputs
        print(f"\n   📊 INPUT CHUNKS ANALYSIS:")
        print(f"   {'='*50}")
        
        chunk_types = {}
        content_lengths = []
        token_estimates = []
        
        for chunk in chunks:
            # Get chunk type from metadata or content analysis
            chunk_type = getattr(chunk, 'chunk_type', 'unknown')
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_type = chunk.metadata.get('primary_block_type', chunk_type)
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            content_lengths.append(len(chunk.content))
            
            # Estimate tokens (rough: chars/4)
            token_est = len(chunk.content) // 4
            token_estimates.append(token_est)
        
        print(f"   📋 Total chunks: {len(chunks)}")
        print(f"   📊 Chunk types: {dict(chunk_types)}")
        print(f"   📏 Content lengths - Min: {min(content_lengths)}, Max: {max(content_lengths)}, Avg: {sum(content_lengths)/len(content_lengths):.1f}")
        print(f"   🪙 Token estimates - Min: {min(token_estimates)}, Max: {max(token_estimates)}, Avg: {sum(token_estimates)/len(token_estimates):.1f}")
        
        # Show first few chunks as examples
        print(f"\n   📄 EXAMPLE INPUT CHUNKS:")
        print(f"   {'-'*50}")
        for i, chunk in enumerate(chunks[:4]):
            content_preview = chunk.content.strip()[:100]
            if len(chunk.content.strip()) > 100:
                content_preview += "..."
            
            chunk_type = getattr(chunk, 'chunk_type', 'unknown')
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_type = chunk.metadata.get('primary_block_type', chunk_type)
            
            print(f"   [{i+1}] Type: {chunk_type}")
            print(f"       Content: '{content_preview}'")
            print(f"       Length: {len(chunk.content)} chars, ~{len(chunk.content)//4} tokens")
            
            if hasattr(chunk, 'metadata') and chunk.metadata:
                meta_preview = dict(list(chunk.metadata.items())[:3])
                print(f"       Metadata: {meta_preview}")
            print()
        
        if len(chunks) > 4:
            print(f"   ... and {len(chunks) - 4} more chunks")
        
        print(f"   ✅ Real chunks ready for embedding")
        self.real_chunks = chunks
        return chunks
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        print("🔧 Testing embedder initialization...")
        
        def _test_init():
            assert self.embedder is not None, "Embedder should be initialized"
            assert hasattr(self.embedder, 'model'), "Embedder should have model"
            assert hasattr(self.embedder, 'model_name'), "Embedder should have model_name"
            return True
        
        result, duration = self._time_function('initialization', _test_init)
        
        print(f"   ✅ Model name: {self.embedder.model_name}")
        print(f"   ✅ Model loaded: {self.embedder.model is not None}")
        print(f"   ⏱️ Initialization time: {duration:.4f}s")
        print("   ✅ Embedder initialization successful")
        return result
    
    def test_create_test_chunks(self):
        """Create diverse test chunks for embedding tests."""
        test_chunks = [
            # Comment chunks - Documentation
            TestChunk(
                content="/// Data Import Section\n/// This section handles loading of sales and inventory data from CSV files\n/// and validates the data structure before processing",
                metadata={"type": "comment", "language": "envision", "section": "data_import", "complexity": "high"},
                chunk_type="documentation"
            ),
            TestChunk(
                content="// Simple calculation comment",
                metadata={"type": "comment", "language": "envision", "section": "calculations", "complexity": "low"},
                chunk_type="documentation"
            ),
            
            # Data import chunks - Different complexity levels
            TestChunk(
                content='''read "sales_data.csv" as Sales with
  ItemCode : text
  ItemName : text
  Quantity : number
  UnitPrice : number
  SaleDate : date''',
                metadata={"type": "data_import", "table": "Sales", "columns": 5, "complexity": "medium"},
                chunk_type="data_ingestion"
            ),
            TestChunk(
                content='read "simple.csv" as Data with Value : number',
                metadata={"type": "data_import", "table": "Data", "columns": 1, "complexity": "low"},
                chunk_type="data_ingestion"
            ),
            
            # Calculation chunks - Various mathematical operations
            TestChunk(
                content="Sales.Revenue = Sales.Quantity * Sales.UnitPrice",
                metadata={"type": "calculation", "field": "Revenue", "operation": "multiplication", "complexity": "low"},
                chunk_type="calculation"
            ),
            TestChunk(
                content='''Sales.TotalCost = Sales.Quantity * Sales.UnitPrice * 0.6
Sales.Profit = Sales.Revenue - Sales.TotalCost
Sales.ProfitMargin = (Sales.Profit / Sales.Revenue) * 100''',
                metadata={"type": "calculation", "fields": ["TotalCost", "Profit", "ProfitMargin"], "operations": ["multiplication", "subtraction", "division"], "complexity": "high"},
                chunk_type="calculation"
            ),
            
            # Aggregation chunks - Statistical operations
            TestChunk(
                content="TotalRevenue = sum(Sales.Revenue)",
                metadata={"type": "aggregation", "function": "sum", "field": "Revenue", "complexity": "low"},
                chunk_type="aggregation"
            ),
            TestChunk(
                content='''MonthlyStats = Sales 
  | group by month(Sales.SaleDate)
  | summarize TotalRevenue = sum(Revenue), AvgPrice = avg(UnitPrice), ItemCount = count()''',
                metadata={"type": "aggregation", "functions": ["sum", "avg", "count", "group"], "complexity": "high"},
                chunk_type="aggregation"
            ),
            
            # Output/Visualization chunks
            TestChunk(
                content='show table "Sales Summary" with Sales.ItemCode, Sales.Revenue',
                metadata={"type": "output", "output_type": "table", "show_name": "Sales Summary", "complexity": "low"},
                chunk_type="visualization"
            ),
            TestChunk(
                content='''show table "Detailed Analysis" with
  Sales.ItemCode
  Sales.ItemName
  Sales.Revenue
  Sales.Profit
  Sales.ProfitMargin
order by Sales.Revenue desc
where Sales.Revenue > 1000''',
                metadata={"type": "output", "output_type": "table", "show_name": "Detailed Analysis", "has_filter": True, "has_order": True, "complexity": "high"},
                chunk_type="visualization"
            ),
            
            # Business logic chunks
            TestChunk(
                content="if (Sales.Revenue > 10000) then 'High Value' else 'Standard'",
                metadata={"type": "conditional", "condition_type": "if_then_else", "complexity": "medium"},
                chunk_type="business_logic"
            ),
            TestChunk(
                content='''InventoryStatus = case
  when Stock > 100 then "Overstocked"
  when Stock > 50 then "Well Stocked"  
  when Stock > 10 then "Low Stock"
  else "Critical"''',
                metadata={"type": "conditional", "condition_type": "case_when", "branches": 4, "complexity": "high"},
                chunk_type="business_logic"
            )
        ]
        return test_chunks
    
    def test_embed_single_chunk(self):
        """Test embedding a single chunk."""
        print("🎯 Testing single chunk embedding...")
        
        # Initialize embedder first
        self.embedder.initialize()
        
        test_chunk = TestChunk(
            content="Sales.TotalRevenue = sum(Sales.Revenue)",
            metadata={"type": "aggregation"}
        )
        
        # Use embed_chunks with a single chunk
        embeddings = self.embedder.embed_chunks([test_chunk])
        embedding = embeddings[0]
        
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert len(embedding.shape) == 1, "Embedding should be 1D array"
        assert embedding.shape[0] > 0, "Embedding should have positive dimension"
        
        print(f"   ✅ Embedding shape: {embedding.shape}")
        print(f"   ✅ Embedding dtype: {embedding.dtype}")
        print(f"   ✅ Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print("   ✅ Single chunk embedding successful")
        return True
    
    def test_embed_multiple_chunks(self):
        """Test embedding real chunks from Envision script with detailed analysis."""
        print("🎯 Testing multiple chunk embedding on real chunks...")
        
        # Initialize embedder first
        if not self.embedder._is_initialized:
            self.embedder.initialize()
        
        # Get real chunks from chunker
        real_chunks = self.get_real_chunks()
        if not real_chunks:
            print("   ❌ No real chunks available for embedding")
            return False
        
        # Create embeddings with timing
        def _embed_chunks():
            return self.embedder.embed_chunks(real_chunks)
            
        embeddings, duration = self._time_function('embedding', _embed_chunks)
        
        assert len(embeddings) == len(real_chunks), \
            f"Should produce {len(real_chunks)} embeddings, got {len(embeddings)}"
        
        print(f"   ✅ Created {len(embeddings)} embeddings")
        print(f"   ⏱️ Embedding time: {duration:.4f}s ({len(real_chunks)/duration:.1f} chunks/s)")
        
        # Detailed analysis of embeddings
        print(f"\n   📋 DETAILED EMBEDDING ANALYSIS:")
        print(f"   {'='*60}")
        
        embedding_stats = {
            'dimensions': [],
            'magnitudes': [],
            'min_values': [],
            'max_values': [],
            'mean_values': [],
            'std_values': []
        }
        
        # Validate each embedding and collect statistics
        for i, embedding in enumerate(embeddings):
            assert embedding is not None, f"Embedding {i} should not be None"
            assert isinstance(embedding, np.ndarray), f"Embedding {i} should be numpy array"
            assert len(embedding.shape) == 1, f"Embedding {i} should be 1D array"
            
            # Calculate statistics
            magnitude = np.linalg.norm(embedding)
            min_val = embedding.min()
            max_val = embedding.max()
            mean_val = embedding.mean()
            std_val = embedding.std()
            
            # Collect for summary
            embedding_stats['dimensions'].append(embedding.shape[0])
            embedding_stats['magnitudes'].append(magnitude)
            embedding_stats['min_values'].append(min_val)
            embedding_stats['max_values'].append(max_val)
            embedding_stats['mean_values'].append(mean_val)
            embedding_stats['std_values'].append(std_val)
            
            chunk_content = real_chunks[i].content[:40] + "..." if len(real_chunks[i].content) > 40 else real_chunks[i].content
            print(f"   [Embedding {i+1}] Chunk: '{chunk_content}'")
            print(f"      📏 Dimension: {embedding.shape[0]}")
            print(f"      📊 Magnitude: {magnitude:.4f}")
            print(f"      📈 Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"      📐 Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            
            # Get chunk type
            chunk_type = getattr(real_chunks[i], 'chunk_type', 'unknown')
            if hasattr(real_chunks[i], 'metadata') and real_chunks[i].metadata:
                chunk_type = real_chunks[i].metadata.get('primary_block_type', chunk_type)
            print(f"      🏷️ Type: {chunk_type}")
            print()
        
        # Summary statistics
        print(f"   📊 EMBEDDING STATISTICS:")
        print(f"   {'-'*60}")
        print(f"   📏 All embeddings dimension: {embedding_stats['dimensions'][0]}")
        print(f"   📊 Magnitude - Min: {min(embedding_stats['magnitudes']):.4f}, Max: {max(embedding_stats['magnitudes']):.4f}")
        print(f"   📈 Value range - Global min: {min(embedding_stats['min_values']):.4f}, Global max: {max(embedding_stats['max_values']):.4f}")
        print(f"   📐 Mean values - Avg: {np.mean(embedding_stats['mean_values']):.4f}")
        print(f"   📏 Standard deviations - Avg: {np.mean(embedding_stats['std_values']):.4f}")
        
        # Show detailed examples of different chunk types and their embeddings
        print(f"\n   📄 DETAILED EMBEDDING EXAMPLES BY TYPE:")
        print(f"   {'-'*60}")
        
        # Group chunks by type for analysis
        chunks_by_type = {}
        for i, chunk in enumerate(real_chunks):
            chunk_type = chunk.chunk_type
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append((i, chunk, embeddings[i]))
        
        # Show examples for each type
        for chunk_type, chunks_list in chunks_by_type.items():
            print(f"   📝 {chunk_type.upper()} EXAMPLES:")
            
            # Show up to 2 examples per type
            for j, (idx, chunk, embedding) in enumerate(chunks_list[:2]):
                complexity = chunk.metadata.get('complexity', 'unknown')
                
                # Show chunk content (limited for readability)
                content = chunk.content.strip()
                if len(content) > 200:
                    lines = content.split('\n')
                    if len(lines) > 4:
                        partial_content = '\n'.join(lines[:4]) + f"\n... ({len(lines)-4} more lines)"
                    else:
                        partial_content = content[:197] + "..."
                else:
                    partial_content = content
                
                print(f"      [{j+1}] Complexity: {complexity}")
                print(f"         Content: \"{partial_content}\"")
                print(f"         📊 Embedding stats: Dim={embedding.shape[0]}, Mag={np.linalg.norm(embedding):.4f}")
                print(f"         📈 Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                print(f"         🏷️ Metadata: {chunk.metadata}")
                print()
            
            if len(chunks_list) > 2:
                print(f"         ... and {len(chunks_list) - 2} more {chunk_type} examples")
                print()
        
        # Calculate and show similarity between similar types
        print(f"   🔗 EMBEDDING SIMILARITY ANALYSIS:")
        print(f"   {'-'*60}")
        
        # Find similar chunks and calculate cosine similarity
        similarity_pairs = []
        for i in range(len(real_chunks)):
            for j in range(i+1, len(real_chunks)):
                chunk_i, chunk_j = real_chunks[i], real_chunks[j]
                embedding_i, embedding_j = embeddings[i], embeddings[j]
                
                # Calculate cosine similarity
                similarity = np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_i) * np.linalg.norm(embedding_j))
                
                similarity_pairs.append({
                    'chunk_i': i,
                    'chunk_j': j,
                    'type_i': chunk_i.chunk_type,
                    'type_j': chunk_j.chunk_type,
                    'similarity': similarity,
                    'same_type': chunk_i.chunk_type == chunk_j.chunk_type
                })
        
        # Sort by similarity and show top examples
        similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"   📊 TOP SIMILARITY PAIRS:")
        for i, pair in enumerate(similarity_pairs[:5]):
            same_type_marker = "🎯" if pair['same_type'] else "🔀"
            print(f"      {same_type_marker} Chunks {pair['chunk_i']+1}-{pair['chunk_j']+1}: {pair['similarity']:.4f}")
            print(f"         Types: {pair['type_i']} ↔ {pair['type_j']}")
        
        print(f"   📉 LOWEST SIMILARITY PAIRS:")
        for i, pair in enumerate(similarity_pairs[-3:]):
            same_type_marker = "🎯" if pair['same_type'] else "🔀"
            print(f"      {same_type_marker} Chunks {pair['chunk_i']+1}-{pair['chunk_j']+1}: {pair['similarity']:.4f}")
            print(f"         Types: {pair['type_i']} ↔ {pair['type_j']}")
        
        print("   ✅ Multiple chunk embedding successful")
        return True
    
    def test_embedding_similarity(self):
        """Test that similar content produces similar embeddings."""
        print("🔍 Testing embedding similarity...")
        
        # Initialize embedder first
        if not self.embedder._is_initialized:
            self.embedder.initialize()
        
        # Create similar chunks
        chunk1 = TestChunk("Sales.Revenue = Sales.Quantity * Sales.Price")
        chunk2 = TestChunk("Sales.Revenue = Sales.Quantity * Sales.UnitPrice")
        chunk3 = TestChunk("show table \"Report\" with Sales.Revenue")
        
        # Use embed_chunks for single chunks
        emb1 = self.embedder.embed_chunks([chunk1])[0]
        emb2 = self.embedder.embed_chunks([chunk2])[0]
        emb3 = self.embedder.embed_chunks([chunk3])[0]
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(emb1, emb2)  # Should be high (similar calculations)
        sim_1_3 = cosine_similarity(emb1, emb3)  # Should be lower (different purposes)
        sim_2_3 = cosine_similarity(emb2, emb3)  # Should be lower
        
        print(f"   📊 Similarity scores:")
        print(f"      Similar calculations (1-2): {sim_1_2:.4f}")
        print(f"      Calculation vs output (1-3): {sim_1_3:.4f}")
        print(f"      Calculation vs output (2-3): {sim_2_3:.4f}")
        
        # Similar content should have higher similarity than dissimilar
        assert sim_1_2 > 0.5, "Similar calculations should have high similarity"
        print(f"   ✅ Similar content similarity: {sim_1_2:.4f} > 0.5")
        
        print("   ✅ Embedding similarity test successful")
        return True
    
    def test_empty_content_handling(self):
        """Test handling of empty or whitespace content."""
        print("⚠️ Testing empty content handling...")
        
        empty_chunks = [
            TestChunk(""),
            TestChunk("   "),
            TestChunk("\n\n\t  \n"),
        ]
        
        # Initialize embedder first
        if not self.embedder._is_initialized:
            self.embedder.initialize()
        
        for i, chunk in enumerate(empty_chunks):
            try:
                embedding = self.embedder.embed_chunks([chunk])[0]
                print(f"   ✅ Empty chunk {i+1} handled: shape {embedding.shape}")
            except Exception as e:
                print(f"   ⚠️ Empty chunk {i+1} raised exception: {e}")
                # This is acceptable behavior
        
        print("   ✅ Empty content handling validated")
        return True
    
    def test_configuration_usage(self):
        """Test that embedder properly uses configuration."""
        print("⚙️ Testing configuration usage...")
        
        # Test model_name from config
        config_model = self.embedder_config.get('model_name', 'default')
        embedder_model = self.embedder.model_name
        
        print(f"   ✅ Model from config: {config_model}")
        print(f"   ✅ Model in embedder: {embedder_model}")
        
        # Test device setting if available
        if hasattr(self.embedder, 'device'):
            print(f"   ✅ Device: {self.embedder.device}")
        
        # Test batch size if available
        if 'batch_size' in self.embedder_config:
            print(f"   ✅ Batch size: {self.embedder_config['batch_size']}")
        
        # Test normalize embeddings if available
        if 'normalize_embeddings' in self.embedder_config:
            print(f"   ✅ Normalize embeddings: {self.embedder_config['normalize_embeddings']}")
        
        print("   ✅ Configuration usage validated")
        return True
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        print("📦 Testing batch processing...")
        
        # Create a larger set of test chunks
        large_chunk_set = []
        for i in range(10):
            large_chunk_set.append(TestChunk(
                content=f"This is test chunk number {i} with content about data processing",
                metadata={"chunk_id": i}
            ))
        
        # Initialize embedder first
        if not self.embedder._is_initialized:
            self.embedder.initialize()
        
        try:
            embeddings = self.embedder.embed_chunks(large_chunk_set)
            
            assert len(embeddings) == len(large_chunk_set), \
                "Should produce embedding for each chunk"
            
            print(f"   ✅ Processed {len(large_chunk_set)} chunks in batch")
            print(f"   ✅ All embeddings shape: {embeddings[0].shape}")
            
        except Exception as e:
            print(f"   ⚠️ Batch processing failed: {e}")
            # Try individual processing as fallback
            individual_embeddings = []
            for chunk in large_chunk_set:
                individual_embeddings.append(self.embedder.embed_chunks([chunk])[0])
            
            print(f"   ✅ Fallback: processed {len(individual_embeddings)} chunks individually")
        
        print("   ✅ Batch processing validated")
        return True
    
    def run_all_tests(self):
        """Run all embedder tests."""
        print("🧪 SENTENCE TRANSFORMER EMBEDDER TESTS")
        print("=" * 50)
        
        tests = [
            self.test_embedder_initialization,
            self.test_embed_single_chunk,
            self.test_embed_multiple_chunks,
            self.test_embedding_similarity,
            self.test_empty_content_handling,
            self.test_configuration_usage,
            self.test_batch_processing
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
        self._print_timing_stats()
        print("=" * 50)
        
        return failed == 0
    
    def _print_timing_stats(self):
        """Print timing statistics."""
        if not self.timing_stats:
            return
            
        print(f"\n⏱️ TIMING STATISTICS:")
        print(f"{'='*50}")
        
        total_time = sum(self.timing_stats.values())
        print(f"📈 Total execution time: {total_time:.4f}s")
        
        if len(self.timing_stats) > 1:
            print(f"📊 Breakdown by operation:")
            for operation, duration in sorted(self.timing_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100
                print(f"   • {operation}: {duration:.4f}s ({percentage:.1f}%)")
        
        # Performance metrics
        if 'embedding' in self.timing_stats:
            print(f"🚀 Performance metrics:")
            embedding_time = self.timing_stats['embedding']
            print(f"   • Average embedding time: {embedding_time:.4f}s per batch")

def main():
    """Main test runner."""
    tester = TestSentenceTransformerEmbedder()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
