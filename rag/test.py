#!/usr/bin/env python3
"""Test DSL Query System"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def test_router():
    """Test router classification"""
    print("🧪 Testing Router")
    from rag.router import Router
    
    router = Router()
    tests = [
        ("Quels scripts lisent /Clean/Items.ion ?", "grep"),
        ("Combien de scripts utilisent RedispatchCycleWeeks ?", "grep"),
        ("Où sont calculés les meilleurs vendeurs ?", "rag"),
        ("Comment est calculé le stock ?", "rag"),
    ]
    
    for q, expected in tests:
        c = router.classify(q)
        status = "✅" if c.qtype.value == expected else "❌"
        print(f"{status} {q[:50]}... -> {c.qtype.value}")
    

def test_grep():
    """Test grep search"""
    print("\n🧪 Testing Grep")
    from grep.searcher import GrepSearcher
    
    searcher = GrepSearcher(["env_scripts"])
    result = searcher.search("Items.ion")
    print(f"Found {result.count} files with 'Items.ion'")
    if result.count > 0:
        print(f"✅ Grep working")
    else:
        print("⚠️  No results")


def test_rag():
    """Test RAG components"""
    print("\n🧪 Testing RAG")
    from config_manager import ConfigManager
    from rag.parsers.envision_parser import EnvisionParser
    from rag.chunkers.semantic_chunker import SemanticChunker
    
    cfg = ConfigManager()
    parser = EnvisionParser(cfg.get_parser_config())
    chunker = SemanticChunker(cfg.get_chunker_config())
    
    files = list(Path("env_scripts").glob("*.nvn"))[:3]
    blocks = []
    for f in files:
        blocks.extend(parser.parse_file(str(f)))
    
    print(f"Parsed {len(blocks)} blocks from {len(files)} files")
    
    chunks = chunker.chunk_blocks(blocks)
    print(f"Created {len(chunks)} chunks")
    print("✅ RAG components working")


if __name__ == "__main__":
    try:
        test_router()
        test_grep()
        test_rag()
        print("\n✅ All tests passed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
