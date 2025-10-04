"""
Simple test to verify the preprocessing structure works.
"""

# No environment setup needed (dev by default)
try:
    from preprocessing.parsers.envision_parser import EnvisionParser
    from preprocessing.chunkers.semantic_chunker import SemanticChunker
    from preprocessing.core.base_parser import CodeBlock
    from config_manager import get_config
    
    print("✅ All imports successful!")
    
    # Test configuration loading
    config = get_config()
    print(f"✅ Configuration loaded: {config}")
    
    # Test parser
    parser = EnvisionParser()
    print(f"✅ Parser created: {type(parser).__name__}")
    print(f"   Supported extensions: {parser.supported_extensions}")
    
    # Test chunker
    chunker = SemanticChunker()
    print(f"✅ Chunker created: {type(chunker).__name__}")
    print(f"   Max tokens: {chunker.max_chunk_tokens}")
    print(f"   Overlap lines: {chunker.overlap_lines}")
    print(f"   Chars per token: {chunker.chars_per_token}")
    
    # Test parsing simple content
    content = """/// Test comment
const projectFolder = ""
read "path" as Catalog with
  ItemCode : text
  ImageName : text

Catalog.Sku = concat(Catalog.ItemCode,"-",Catalog.ItemLoc)

show table "Test" with
  Catalog.Sku"""
    
    blocks = parser.parse_content(content, "test.nvn")
    print(f"✅ Parsed {len(blocks)} blocks from test content")
    
    for i, block in enumerate(blocks):
        print(f"   {i+1}. {block.block_type}: {block.name or 'unnamed'} (lines {block.line_start}-{block.line_end})")
    
    # Test chunking
    chunks = chunker.chunk_blocks(blocks)
    print(f"✅ Created {len(chunks)} chunks from blocks")
    
    for i, chunk in enumerate(chunks):
        print(f"   {i+1}. {chunk.chunk_type}: {chunk.size_tokens} tokens, {len(chunk.original_blocks)} blocks")
    
    print("\n🎉 Basic preprocessing structure test successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is expected if dependencies are not installed yet.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()