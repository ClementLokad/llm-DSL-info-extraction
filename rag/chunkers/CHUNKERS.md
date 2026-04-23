# ✂️ Chunkers - Semantic Envision Segmentation

> Transform parsed Envision code blocks into fixed-size semantic chunks optimized for RAG embeddings and retrieval.

## 📁 Folder Contents

- **`envision_chunker.py`** - Envision DSL chunker (production)
- **`semantic_chunker.py`** - Alternative semantic chunker (tests)

---

## 🎯 EnvisionChunker (Production)

Divides Envision code blocks into fixed-size chunks (e.g., 512 tokens) with contextual overlap.

### How it Works

```
Input: 6078 parsed blocks (all Envision scripts from project)
         ↓
    Split large blocks (if > max_tokens)
         ↓
    Chunk with token limit
         ↓
    Add overlap between adjacent chunks
         ↓
Output: 1084 optimized chunks for RAG (~5.6 blocks/chunk ratio)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Token limit** | `max_chunk_tokens` configurable (default: 512) |
| **Overlap** | `overlap_lines` (default: 3) for context |
| **Section respect** | Section headers (`///=====`) = new chunks |
| **Adaptive split** | Large blocks split with internal overlap |
| **Metadata** | Section, dependencies, token count, line range |
| **Backtracking** | Small chunks "filled" with previous lines |

### Metadata per Chunk

```python
chunk.metadata = {
    'section': 'DATA LOADING',                    # Script section
    'has_overlap': True,                          # Overlap with previous chunk
    'overlap_with_chunk': 2,                      # Previous chunk ID
    'overlap_blocks': 3,                          # Number of overlapped blocks
    'external_dependencies': {'Orders', 'Items'}, # External dependencies
    'dependency_providers': {                     # Who provides each dependency
        'Orders': 0, 
        'Items': 1
    },
    'token_count': 485,
    'block_types': ['read', 'assignment', 'show'],
    'line_range': (45, 89),
    'file_path': '/path/to/script.nvn'
}
```

### Simple Usage

```python
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.parsers.envision_parser import EnvisionParser

# Parse first
parser = EnvisionParser()
blocks = parser.parse_file("script.nvn")

# Then chunk
chunker = EnvisionChunker(config)
chunks = chunker.chunk_blocks(blocks)

# Result
for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.size_tokens} tokens, section='{chunk.metadata['section']}'")
```

### Utility Function

```python
from rag.chunkers.envision_chunker import parse_and_chunk_file

# One-liner: parse + chunk
blocks, chunks = parse_and_chunk_file("script.nvn")
```

---

## 📊 Configuration

Via `config.yaml` (or Python dict):

```yaml
chunking:
  max_chunk_tokens: 512      # Max tokens per chunk
  overlap_lines: 3            # Lines of overlap
  preserve_boundaries: true   # Respect section headers
```

---

## 🧵 SemanticChunker (Alternative)

Semantic grouping by sections and dependencies. Less used than `EnvisionChunker`.

**Used for** : Tests, alternative explorations  
**Strategies** : `group_by_section`, `group_related_assignments`, `keep_read_statements_separate`

---

## 📍 Real Usage in Project

### Index Building

**`build_index.py`** - Standard FAISS index
```python
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.parsers.envision_parser import EnvisionParser

parser = EnvisionParser()
chunker = EnvisionChunker(cfg.get_chunker_config())

for script_file in script_files:
    blocks = parser.parse_file(script_file)
    chunks = chunker.chunk_blocks(blocks, start_id=len(chunks))
    # → Chunks sent to embedders + FAISS index
```

**`build_summary_index.py`** - Summary index (grouped by sections)
```python
chunks = chunker.chunk_blocks(current_blocks, start_id=len(chunks))
# → Chunks summarized + FAISS index
```

**`build_raptor_index.py`** - Hierarchical RAPTOR index
```python
chunks = chunker.chunk_blocks(current_blocks)
# → Chunks → Embedding → RAPTOR index
```

### Complete RAG Workflow

1. **Parsing** → `EnvisionParser.parse_file()` = List[CodeBlock]
2. **Chunking** → `EnvisionChunker.chunk_blocks()` = List[CodeChunk]
3. **Embedding** → Vectorize chunk.content
4. **Indexing** → Store in FAISS/Raptor/Qdrant with metadata
5. **Retrieval** → Query → Top-k chunks → LLM context

---

## 🏗️ Architecture

All chunkers inherit from `BaseChunker` and produce `CodeChunk`.

### DataClasses

```python
# Input (after parsing)
@dataclass
class CodeBlock:
    content: str
    block_type: BlockType          # IMPORT, READ, ASSIGNMENT, etc.
    name: str                      # Block identifier
    line_start: int                # Start line (1-indexed)
    line_end: int
    dependencies: Set[str]         # Used dependencies
    definitions: Set[str]          # Defined by this block
    metadata: Dict                 # Parser-specific metadata

# Output
@dataclass
class CodeChunk:
    content: str                   # Chunk content
    chunk_id: int                  # Unique ID
    original_blocks: List[CodeBlock]  # Blocks composing this chunk
    size_tokens: int               # Estimated tokens
    dependencies: Set[str]         # Dependencies (union of blocks)
    definitions: Set[str]          # Definitions (union of blocks)
    metadata: Dict                 # Section, overlap, providers, etc.
```



## 📦 Dependencies

**Internal**:
- `rag.core.base_chunker` - Abstract base
- `rag.parsers.envision_parser` - Parsing
- `config_manager` - Configuration

**External**:
- `tiktoken` (optional) - Exact token counting

---

## 🔧 API Reference

### EnvisionChunker

```python
class EnvisionChunker(BaseChunker):
    def __init__(self, config: Dict[str, Any])
    def chunk_blocks(self, code_blocks: List[CodeBlock], 
                     start_id: int = 0) -> List[CodeChunk]
```

**Config Parameters**:
- `max_chunk_tokens` (int) - Token limit per chunk
- `overlap_lines` (int) - Overlap lines between chunks

### CodeChunk Methods

```python
chunk.get_line_range() -> Tuple[int, int]       # (start, end)
chunk.get_token_count() -> int                  # Chunk tokens
chunk.add_block(block, max_tokens) -> bool      # Add a block
```

---

## 📋 Common Issues

| Issue | Solution |
|----------|----------|
| Chunks too small | Increase `max_chunk_tokens` |
| Loss of context | Increase `overlap_lines` |
| Bad split for large blocks | Check `has_split_overlap` metadata |

---

## 📚 Related Resources

- [Parser Documentation](../parsers/PARSER.md) - How blocks are extracted
- [RAG Pipeline](../RAG.md) - Complete pipeline parsing → embedding → indexing
- [Configuration](../../config.yaml) - Global parameters

## 🔗 Related Components

- [Embedders Documentation](../embedders/EMBEDDERS.md) - Embedding chunks after chunking
- [Retrievers Documentation](../retrievers/RETRIEVERS.md) - Retrieving chunks from index
