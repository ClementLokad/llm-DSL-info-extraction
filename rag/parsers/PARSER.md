# 📝 Parser - Semantic Envision Extraction

> Parse Envision scripts into semantic blocks with complete dependency tracking.

## 📁 Folder Contents

- **`envision_parser.py`** - Envision DSL parser (production)
- **`old_envision_parser.py`** - Previous version (archive)

---

## 🎯 EnvisionParser (Production)

Analyzes Envision scripts line by line and extracts semantic blocks with dependencies.

### How it Works

```
Input: Envision script (.nvn)
         ↓
    Identify block types (READ, ASSIGNMENT, etc.)
         ↓
    Extract dependencies (Tables/variables used)
         ↓
    Extract definitions (Tables/variables created)
         ↓
Output: 6078 CodeBlock with metadata
```

### Recognized Block Types

| Type | Example | Dependencies | Defines |
|------|---------|-------------|----------| 
| `SECTION_HEADER` | `///===== DATA LOAD =====` | None | Section name |
| `IMPORT` | `import "/path" as Tbl` | Path variables | `Tbl` |
| `READ` | `read "file.ion" as Items` | Path variables | `Items` |
| `WRITE` | `write Items into "out.ion"` | `Items`, paths | None |
| `CONST` | `const path = "/data"` | Variables RHS | `path` |
| `EXPORT` | `export table Result = ...` | Tables RHS | `Result` |
| `TABLE_DEFINITION` | `table Sales = cross(Items, Week)` | `Items`, `Week` | `Sales` |
| `ASSIGNMENT` | `Items.Total = sum(Orders.Amount)` | `Items`, `Orders` | `Items` (field) |
| `SHOW` | `show table "Results"` | Tables referenced | None |
| `KEEP_WHERE` | `keep where Orders.IsValid` | `Orders` | None |
| `COMMENT` | `// Comment` | None | None |
| `FORM_READ` | `read form with field : text` | None | None |

### Dependency Extraction (Algorithm)

**Goal**: Avoid false positives while capturing true dependencies

```python
# ❌ WRONG - Do not extract
Items.Price = sum(Orders.Amount)
# → Dependencies: {Orders}  (NOT {Items, Price, Orders, Amount})

const path = "/Clean/Data" 
# → Dependencies: {}  (NOT {Clean, Data})

areValid = all(Items.Stock >= 0)
# → Dependencies: {Items}  (NOT {Stock})

# ✅ CORRECT - Extract
show table "Sales" {unit: #(Region)}
# → Dependencies: {Region}

read "\{inputFolder}Items.ion" as Items
# → Dependencies: {inputFolder}

total = Items.Cost * markup
# → Dependencies: {Items, markup}
```

**Techniques**:
- Strip strings before extraction
- Table.field → extract "Table" only
- Filter built-ins (sum, max, where, etc.)
- Context check (skip if defined, skip if in comment)
- Length check (>=2 chars for standalone identifiers)

### Simple Usage

```python
from rag.parsers.envision_parser import EnvisionParser

# Parser
parser = EnvisionParser()
blocks = parser.parse_file("script.nvn")

# Results
for block in blocks:
    print(f"{block.block_type.value}: '{block.name}'")
    print(f"  Lines: {block.line_start}-{block.line_end}")
    print(f"  Dependencies: {block.dependencies}")
    print(f"  Defines: {block.definitions}")
```

### Metadata per Block

```python
block = CodeBlock(
    content: str                      # Source code
    block_type: BlockType             # IMPORT, READ, ASSIGNMENT, etc.
    name: Optional[str]               # Identifier (table/var name)
    line_start: int                   # Start line (1-indexed)
    line_end: int                     # End line
    file_path: str                    # /path/to/script.nvn
    dependencies: Set[str]            # Tables/variables used
    definitions: Set[str]             # Tables/variables created
    metadata: Dict                    # Parser-specific data
)
```

---

## 📊 Configuration

Via `config.yaml` :

```yaml
parser:
  supported_extensions:
    - .nvn
```

---

## 🔧 API Reference

### EnvisionParser

```python
class EnvisionParser(BaseParser):
    def __init__(self, config: Dict[str, Any] = None)
    def parse_file(self, file_path: str) -> List[CodeBlock]
    def parse_content(self, content: str, file_path: str = "") -> List[CodeBlock]
```

### CodeBlock

```python
# Properties
block.content: str
block.block_type: BlockType
block.name: Optional[str]
block.line_start: int
block.line_end: int
block.dependencies: Set[str]
block.definitions: Set[str]

# Methods
block.get_token_count() -> int
block.to_dict() -> Dict
CodeBlock.from_dict(data: Dict) -> CodeBlock
len(block) -> int  # number of lines
```

### BlockType Enum

```python
BlockType.COMMENT
BlockType.SECTION_HEADER
BlockType.IMPORT
BlockType.READ
BlockType.WRITE
BlockType.CONST
BlockType.EXPORT
BlockType.TABLE_DEFINITION
BlockType.ASSIGNMENT
BlockType.SHOW
BlockType.KEEP_WHERE
BlockType.FORM_READ
BlockType.CONTROL_FLOW
BlockType.UNKNOWN
```

---

## 📍 Real Usage in Project

### Index Building

All `build_*.py` scripts use the parser:

```python
from rag.parsers.envision_parser import EnvisionParser

parser = EnvisionParser()
for script_file in script_files:
    blocks = parser.parse_file(script_file)
    # → 6078 blocks produced total
    # → Sent to chunker
```

### Complete RAG Workflow

1. **Parsing** → `EnvisionParser.parse_file()` = **6078 CodeBlock**
2. **Chunking** → `EnvisionChunker.chunk_blocks()` = 1084 CodeChunk
3. **Embedding** → Vectorize chunk.content
4. **Indexing** → Store in FAISS/Raptor/Qdrant
5. **Retrieval** → Query → Top-k chunks → LLM context

---

## 🏗️ Internal Architecture

### Parsing Strategy

1. **Lexical scan**: Reads script line by line
2. **Type detection**: Regex patterns to identify type (`PATTERNS` dict)
3. **Multi-line handling**: Groups statements across multiple lines via parentheses/indentation
4. **Dependency extraction**: 
   - Table.field → extract "Table"
   - Interpolated variables (#(Var))
   - Paths with variables (\{Var})
5. **Definition tracking**: LHS assignments and imports

### Recognized Patterns (via regex)

```python
PATTERNS = {
    'section_header': r'^///\s*[~=\-]{3,}',
    'comment': r'^///|^//',
    'import': r'^\s*import\s+',
    'read': r'^\s*read\s+',
    'write': r'^\s*write\s+',
    'const': r'^\s*const\s+',
    'export': r'^\s*export\s+',
    'table': r'^\s*table\s+',
    'show': r'^\s*show\s+',
    'keep': r'^\s*keep\s+',
    'where': r'^\s*where\s+',
    'form_read': r'^\s*read\s+form\s+',
}
```

### Filtered Built-ins

`BUILTINS` set contains ~90+ Envision functions (sum, max, cross, etc.) to avoid false positive dependencies.

---

## 📋 Common Issues

| Issue | Solution |
|----------|----------|
| Missing dependencies | Check they're not in `BUILTINS` or `COMMON_PROPERTIES` |
| False positives (field names) | Algorithm already filters via `Table.field` patterns |
| Incorrectly categorized blocks | Check pattern order in `_parse_block()` |
| Performance | ~1000+ lines/second, O(n) complexity |

---

## 📚 Related Resources

### RAG Pipeline Flow
- [RAG Pipeline Overview](../RAG.md) - Complete parsing → chunking → embedding → retrieval flow
- [Chunker Documentation](../chunkers/CHUNKERS.md) - How to chunk parsed blocks

### Related Components
- [Embedders Documentation](../embedders/EMBEDDERS.md) - Embedding chunks from parser output
- [Retrievers Documentation](../retrievers/RETRIEVERS.md) - Retrieving parsed and chunked code
- [Query Transformers Documentation](../query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement before search

### Configuration & Usage
- [Configuration](../../config.yaml) - Global parameters
- [Quick Start Tutorial](../../TUTORIAL.md) - Setup and usage

---

**Built for production RAG applications** 🚀
