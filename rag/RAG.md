# 🎯 RAG Pipeline - Retrieval-Augmented Generation for Envision DSL

> Production RAG system for semantic code retrieval and question answering over Envision DSL codebase (6078 code blocks → 1084 semantic chunks).

## 📊 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER QUESTION / QUERY                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Query Transformation (Optional)│
        │ HyDE: Generate hypothetical docs │
        │ Fusion: Query expansion variants │
        └──────────────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │         Embedding Layer          │
        │  - Dense (all-MiniLM-L6-v2)      │
        │  - Sparse (Qdrant/bm25)          │
        │  - Embedding Dimension: 384      │
        └──────────────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │     Hybrid Retrieval (Qdrant)    │
        │  - Dense search (semantic)       │
        │  - Sparse search (keywords)      │
        │  - RRF Fusion (combine results)  │
        └──────────────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Retrieved Code Chunks (Top-K)  │
        │      + Metadata + Dependencies   │
        └──────────────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  LLM Generation + Context        │
        │  Answer with retrieved context   │
        └──────────────────┬───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GENERATED ANSWER                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ RAG Components

### 1. **Parsers** (`rag/parsers/`)

**Role**: Read and parse Envision scripts (.nvn) into structured CodeBlocks.

**Features**:
- **EnvisionParser**: 12 block types (IMPORT, READ, WRITE, CONST, TABLE_DEFINITION, ASSIGNMENT, etc.)
- **Dependency extraction**: Analyze FKs, cross() links, table references
- **False-positive elimination**: Filter BUILTINS (90+ Envision functions)

**Documentation**: [parsers/PARSER.md](parsers/PARSER.md)

**Production Statistics**:
- **Input**: 6078 code blocks parsed from all scripts
- **Block types**: 12 types with frequency distribution
- **Dependency metadata**: Extracted per block for graph construction

```python
# Usage
parser = EnvisionParser()
code_blocks = parser.parse_script(filepath)  # List[CodeBlock]
```

---

### 2. **Chunkers** (`rag/chunkers/`)

**Role**: Group CodeBlocks into semantic chunks with overlap.

**Features**:
- **Semantic chunking**: Respects section boundaries (doesn't split within a section)
- **Size constraints**: `max_chunk_tokens=512` (default)
- **Overlap**: 3 lines between chunks (default) for context continuity
- **Metadata preservation**: external_dependencies, dependency_providers

**Documentation**: [chunkers/CHUNKERS.md](chunkers/CHUNKERS.md)

**Production Statistics**:
- **Output**: 1084 semantic chunks from 6078 blocks
- **Ratio**: ~5.6 blocks per chunk
- **Overhead**: Chunk-level metadata includes parsed dependencies

```python
# Usage
chunker = EnvisionChunker(max_chunk_tokens=512, overlap_lines=3)
chunks = chunker.chunk(code_blocks)  # List[CodeChunk]
```

---

### 3. **Embedders** (`rag/embedders/`)

**Role**: Convert chunks into vectors (dense + sparse) for retrieval.

**Architecture**:
- **Dense embedder**: all-MiniLM-L6-v2 (384-dim, semantic understanding)
- **Sparse embedder**: Qdrant/bm25 (keyword/syntax matching)
- **Hybrid strategy**: Two complementary vector spaces

**Implementations**:
- **QdrantEmbedder** (Production): Hybrid dense+sparse via Qdrant API
- **SentenceTransformerEmbedder**: Local fastembeds with GPU support
- **GeminiEmbedder / OpenAIEmbedder**: API-based alternatives

**Documentation**: [embedders/EMBEDDERS.md](embedders/EMBEDDERS.md)

**Production Configuration**:
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`
- Sparse model: `Qdrant/bm25`
- Embedding dimension: 384
- Output: Collections "codebase_rag" + "codebase_rag_summary"

```python
# Usage
embedder = QdrantEmbedder(
    dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
    sparse_model_name="Qdrant/bm25",
    embedding_dimension=384
)
embedder.initialize()
embedder.add_chunks(chunks)  # Store in Qdrant
```

---

### 4. **Retrievers** (`rag/retrievers/`)

**Role**: Search and fuse results for hybrid retrieval.

**Implementations**:
- **QdrantRetriever** (Production): Dense + Sparse search with Reciprocal Rank Fusion
- **FAISSRetriever**: In-memory vector DB (HNSW, IVFFlat indexes)

**RRF Strategy**:
```
1. Dense search    → [chunk_a (rank=1), chunk_b (rank=2), ...]
2. Sparse search   → [chunk_c (rank=1), chunk_a (rank=3), ...]
3. RRF Fusion      → Combined ranking with score = 1/(k+rank)
```

**Documentation**: [retrievers/RETRIEVERS.md](retrievers/RETRIEVERS.md)

**Performance production**:
- Qdrant hybrid: ~100-200ms per query
- FAISS: ~20-50ms per query
- RRF fusion overhead: ~5-10ms

```python
# Usage
retriever = QdrantRetriever(collection_name="codebase_rag")
retriever.initialize()

# Search
results = retriever.search(
    query_embedding_dense,
    query_embedding_sparse,
    top_k=5
)  # List[RetrievalResult]
```

---

### 5. **Query Transformers** (`rag/query_transformers/`)

**Role**: Enhance queries via LLM before retrieval.

**Transformers**:
- **HyDE**: Generate N technical summaries answering the query
  - Best for: Concept expansion, semantic understanding
- **Fusion**: Rewrite query with N terminology variants
  - Best for: Terminology diversity, exact keyword matching

**Documentation**: [query_transformers/QUERY_TRANSFORMERS.md](query_transformers/QUERY_TRANSFORMERS.md)

**Usage Pipeline**:
```
Query → Transform (HyDE/Fusion) → N variants
    ↓
Each variant → Embedding (dense + sparse)
    ↓
Each embedding → Retrieve separately
    ↓
RRF Fusion → Final top-k chunks
```

```python
# Usage
transformer = QueryTransformerFactory.create(config)

# HyDE
hyde_summaries = transformer.transform("Calculate revenue?")
# Result: ["Items.Revenue = ...", "Orders join Items...", ...]

# Fusion
fusion_variants = transformer.transform("Calculate revenue?", mode="fusion")
# Result: ["Calculate revenue?", "Sum price * quantity", "Aggregate monetary value"]
```

---

### 6. **Summarizers** (`rag/summarizers/`)

**Role**: Create LLM summaries of chunks for alternative index and context compression.

**Components**:
- **ChunkSummarizer**: Summarize chunks via LLM before storage
- **Qdrant collection**: "codebase_rag_summary" (summaries + embeddings)
- **Use case**: Improve semantic clustering, reduce context noise

**Usage**:
```python
# Summary index (build_summary_index.py)
summarizer = ChunkSummarizer(agent)
for chunk in chunks:
    summary = summarizer.summarize(chunk)  # LLM-generated ~100 words
    summary_chunks.append(SummaryChunk(summary))
    
# Retrieve from summary index
retriever_summary = QdrantRetriever(collection_name="codebase_rag_summary")
summary_results = retriever_summary.search(query_embedding)
```

---

### 7. **Core** (`rag/core/`)

**Role**: Base architecture abstraction and session management.

**Components**:
- **BaseParser**: Interface for all parsers
- **BaseChunker**: Interface for all chunkers
- **BaseEmbedder**: Interface for all embedders
- **BaseRetriever**: Interface for all retrievers
- **BaseQueryTransformer**: Interface for all transformers
- **Session**: Session management and state tracking

**Usage**:
```python
from rag.core import Session

session = Session(config)
session.initialize()  # Setup parser, chunker, embedder, retriever
results = session.search(query)
```

---

### 8. **Utils** (`rag/utils/`)

**Role**: Cross-cutting utilities for the RAG pipeline.

**Modules**:
- **handle_tokens.py**: Token counting and budget management
- **script_scanner.py**: Scanning scripts for statistics
- **switch_db.py**: Switching between stores (Qdrant local, cloud, FAISS)

**Usage**:
```python
# Token handling
from rag.utils.handle_tokens import count_tokens
token_count = count_tokens(chunk.content)

# Database switching
from rag.utils.switch_db import SwitchDB
db = SwitchDB.switch_to_qdrant()  # or switch_to_faiss()
```

---

## 🔄 Complete Pipeline - Step by Step

### Phase 1: Index Building (Off-line)

```
INPUT: Envision DSL Scripts (.nvn files)
   ↓
[1] PARSING
    - EnvisionParser.parse_script()
    - Output: 6078 CodeBlocks
   ↓
[2] CHUNKING
    - EnvisionChunker.chunk()
    - Respect section boundaries
    - Add 3-line overlap
    - Output: 1084 CodeChunks
   ↓
[3] EMBEDDING (Dense + Sparse)
    - QdrantEmbedder.embed_chunks()
    - Dense: all-MiniLM-L6-v2 (384-dim)
    - Sparse: Qdrant/bm25
    - Output: 1084 embedded chunks
   ↓
[4] STORAGE
    - QdrantRetriever.add_chunks()
    - Store in "codebase_rag" collection
    - Persist to disk/cloud
   ↓
[5] SUMMARIZATION (Optional)
    - ChunkSummarizer.summarize() for each chunk
    - Store LLM summaries in "codebase_rag_summary"
   ↓
OUTPUT: Indexed Qdrant database ready for queries
```

### Phase 2: Query & Retrieval (On-line)

```
INPUT: User Query
   ↓
[1] QUERY TRANSFORMATION (Optional)
    Mode: Fusion
    - Generate N query variants with terminology
    - Output: [query_original, query_variant_1, query_variant_2, ...]
   ↓
[2] EMBEDDING QUERIES
    For each query variant:
    - Embed dense: all-MiniLM-L6-v2
    - Embed sparse: Qdrant/bm25
   ↓
[3] HYBRID RETRIEVAL
    QdrantRetriever.search_hybrid():
    
    For each (dense_emb, sparse_emb) pair:
      a) Dense search → Top-N chunks (semantic)
      b) Sparse search → Top-N chunks (keywords)
      c) RRF Fusion:
         score = 1/(k+rank_dense) + 1/(k+rank_sparse)
      d) Top-K chunks from fusion (k=5)
   ↓
[4] CONTEXT AGGREGATION
    - Merge results from all query variants
    - Deduplicate chunks
    - Top-K final chunks (e.g., K=10)
    - Include metadata (dependencies, file info)
   ↓
[5] LLM GENERATION
    - Build context from top-K chunks
    - Send to LLM with system prompt
    - LLM generates answer using retrieved context
   ↓
OUTPUT: Augmented answer with retrieved code context
```

---

## 📈 Production Statistics

| Metric | Value |
|----------|--------|
| **Code blocks** | 6,078 |
| **Semantic chunks** | 1,084 |
| **Block/Chunk ratio** | ~5.6 |
| **Embedding dimension** | 384 |
| **Dense model** | all-MiniLM-L6-v2 |
| **Sparse model** | Qdrant/bm25 |
| **Collections** | 2 (rag + summary) |
| **Retrieval latency (hybrid)** | 100-200ms |
| **Retrieval latency (FAISS)** | 20-50ms |

---

## 🚀 Complete Configuration

```yaml
# config.yaml
parser:
  type: "envision"

chunking:
  type: "envision"
  max_chunk_tokens: 512
  overlap_lines: 3

embedding:
  type: "qdrant"
  embedders:
    dense:
      model_name: "sentence-transformers/all-MiniLM-L6-v2"
      dimension: 384
    sparse:
      model_name: "Qdrant/bm25"
  cache_dir: "data/fastembed_models"

retrieval:
  type: "qdrant"
  collection_name: "codebase_rag"
  qdrant_path: "data/qdrant"
  url: null  # Local (file) if null, cloud if set
  top_k: 10
  k_for_rrf: 60

query_transformer:
  enabled: true
  mode: "fusion"  # "hyde" or "fusion"
  amount_of_generated_instances: "several"

agent:
  type: "deepseek-v3"  # or "sonnet", "deepseek-r1", "mistral", "qwen"
  rate_limit_delay: 0.5
  verbose: false
```

---

## 🔗 Documentation Modules

| Module | Description | Location |
|--------|-------------|----------|
| Parser | Block type identification, dependency extraction | [parsers/PARSER.md](parsers/PARSER.md) |
| Chunker | Semantic chunking with overlap | [chunkers/CHUNKERS.md](chunkers/CHUNKERS.md) |
| Embedder | Hybrid embedding strategy | [embedders/EMBEDDERS.md](embedders/EMBEDDERS.md) |
| Retriever | Vector search with RRF fusion | [retrievers/RETRIEVERS.md](retrievers/RETRIEVERS.md) |
| Query Transformer | HyDE & Fusion query enhancement | [query_transformers/QUERY_TRANSFORMERS.md](query_transformers/QUERY_TRANSFORMERS.md) |
| Summarizer | LLM-based chunk summarization | [summarizers/] |
| Core | Base classes & session management | [core/] |
| Utils | Utilities & helpers | [utils/] |

---

## 📌 Entry Points

### Building Index
```bash
# Full pipeline
python build_index.py

# Summary index
python build_summary_index.py
```

### Query & Retrieval
```python
from rag.core import Session

session = Session(config_path="config.yaml")
session.initialize()

results = session.search(query="Calculate total revenue?")
for result in results:
    print(f"Chunk: {result.content}")
    print(f"Score: {result.score}")
```

---

## 📚 Component Documentation

For detailed information about each RAG component, refer to:

- [Parser Documentation](parsers/PARSER.md) - Envision DSL parsing and block extraction
- [Chunkers Documentation](chunkers/CHUNKERS.md) - Semantic chunking strategies
- [Embedders Documentation](embedders/EMBEDDERS.md) - Dense and sparse embedding models
- [Retrievers Documentation](retrievers/RETRIEVERS.md) - Vector database retrieval with Qdrant/FAISS
- [Query Transformers Documentation](query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement (HyDE, Fusion)

## 🔗 Related Documentation

- [LangGraph Pipeline Orchestration](../pipeline/PIPELINE.md) - How the RAG system is orchestrated
- [Agentic Workflow](../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Multi-tool agent decision loop
- [Quick Start Tutorial](../TUTORIAL.md) - Setup and usage guide

---

**Built for production semantic search and RAG** 🎯