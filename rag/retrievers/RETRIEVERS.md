# 🔍 Retrievers - Semantic and Syntactic Retrieval

> Retrieve relevant code chunks from vector databases via semantic and syntactic search.

## 📁 Folder Contents

- **`faiss_retriever.py`** - Local FAISS index (fast, CPU/GPU)
- **`qdrant_retriever.py`** - Qdrant database (hybrid dense+sparse, production)

---

## 🎯 Retrievers Overview

| Retriever | Type | Backend | Features | Usage |
|-----------|------|---------|------------------|-------|
| **QdrantRetriever** | Hybrid | Qdrant DB | Dense + Sparse, production-ready | **Primary RAG** |
| **FAISSRetriever** | Dense | Local FAISS | Fast, CPU/GPU, multiple indices | Alternative lightweight |

---

## 🚀 QdrantRetriever (Main Production)

Hybrid retriever combining dense (semantic) and sparse (keywords/BM25) search.

### Architecture

```
User Query: "How to calculate Items.Revenue?"
         ↓
    Embed hybrid (dense + sparse)
         ↓
    Search Qdrant collection (2 vector spaces)
         ├─ Dense: Cosine similarity search
         └─ Sparse: BM25 keyword matching
         ↓
    Reciprocal Rank Fusion (RRF)
         ↓
    Top-k results (best of both)
```

### Qdrant Collections

```python
# 2 collections created automatically
collection_name = "codebase_rag"           # Full chunks
collection_name_summary = "codebase_rag_summary"  # LLM summaries
```

Each collection contains 2 vector spaces:
- **Dense**: 384-dim embeddings (semantic)
- **Sparse** : BM25 tokens (keywords + syntax)

### Usage

#### 1. Initialization

```python
from rag.retrievers.qdrant_retriever import QdrantRetriever

config = {
    "collection_name": "codebase_rag",
    "qdrant_path": "./index/qdrant",  # Local storage
    "index_type": "full_chunk"
}

retriever = QdrantRetriever(config)
retriever.initialize(embedding_dimension=384)
```

#### 2. Ingestion (dense-only backward compatible)

```python
# Standard method compatible with FAISS
retriever.add_chunks(chunks, embeddings)  # np.ndarray (N, 384)
```

#### 3. Hybrid Retrieval (new)

```python
# Hybrid search (dense + sparse)
results = retriever.search_hybrid(
    query_text="Calculate total revenue",
    keywords=["Items", "Orders", "Revenue"],
    top_k=5
)

for result in results:
    print(f"Score: {result.score}, File: {result.chunk.metadata['original_file_path']}")
    print(result.chunk.content)
```

### Configuration

```yaml
retriever:
  collection_name: "codebase_rag"
  qdrant_path: "./index/qdrant"          # Local SQLite-like storage
  index_type: "full_chunk"                 # or "summary"
  rerank_multiplier: 2                     # For RRF
  local: true                              # Skip explicit indexing
```

### Advantages

| Advantage | Impact |
|----------|--------|
| **Hybrid search** | Semantic + exact keywords |
| **Production-ready** | Scalable, persistent storage |
| **Local + Cloud** | Local SQLite or remote cloud |
| **Reciprocal Rank Fusion** | Intelligently combine scores |
| **Sparse vectors** | BM25 for syntax/variable names |

---

## ⚡ FAISSRetriever

Local FAISS index for ultra-fast dense search.

### Supported Index Types

```python
index_type: "IndexFlatIP"          # Default - exact inner product
index_type: "IndexIVFFlat"         # IVF quantization (scalable)
index_type: "IndexHNSW"            # Hierarchical Navigable Small World
```

### IVF Configuration

```yaml
retriever:
  faiss:
    index_type: "IndexIVFFlat"
    nlist: 100                     # Number of clusters
    nprobe: 10                     # Clusters to search
```

### HNSW Configuration

```yaml
retriever:
  faiss:
    index_type: "IndexHNSW"
    m: 16                          # Connections per node
    ef_construction: 200           # Build-time parameter
    ef_search: 64                  # Search-time parameter
```

### Usage

```python
from rag.retrievers.faiss_retriever import FAISSRetriever

retriever = FAISSRetriever(config)
retriever.initialize(embedding_dimension=384)

# Add chunks + embeddings
retriever.add_chunks(chunks, embeddings)

# Search
results = retriever.search(query_embedding, top_k=5)
```

### Advantages

| Advantage | Impact |
|----------|--------|
| **Ultra-fast** | No network, local CPU/GPU |
| **Flexible** | Multiple index types |
| **Persistent** | Save/load index |
| **Lightweight** | No external server |

### Disadvantages

- Dense-only search (no sparse/keywords)
- Limited scalability vs Qdrant
- Manual file management



---

## 🏗️ Common Architecture

All retrievers inherit from `BaseRetriever` and return `RetrievalResult`.

```python
class RetrievalResult:
    chunk: CodeChunk           # Retrieved chunk
    score: float              # Relevance score (0-1)
    rank: int                 # Position (1-based)
    metadata: Dict            # Additional metadata
    
    def to_str_for_generation() -> str  # Format for LLM prompt
```

### BaseRetriever Interface

```python
class BaseRetriever(ABC):
    def initialize(self, embedding_dimension: int) -> None: ...
    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[RetrievalResult]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
```

---

## 📍 Real Usage in Project

### Index Building (all `build_*.py`)

```python
from rag.retrievers.qdrant_retriever import QdrantRetriever
from rag.embedders.qdrant_embedder import QdrantEmbedder

embedder = QdrantEmbedder(config)
retriever = QdrantRetriever(config)

embedder.initialize()
retriever.initialize(embedding_dimension=384)

# Ingest chunks + embeddings
for batch in batches:
    embeddings = embedder.embed_chunks(batch)
    retriever.add_chunks(batch, embeddings)
```

### Complete RAG Pipeline

```
1. Parsing    → 6078 blocks
2. Chunking   → 1084 chunks  
3. Embedding  → Dense [1084 × 384] + Sparse BM25
4. Indexing   → Qdrant retriever.add_chunks()
              ↓
5. Query User Question
              ↓
6. Embed Query → embed_text_hybrid(query + keywords)
              ↓
7. Search  → retriever.search_hybrid() → Reciprocal Rank Fusion
              ↓
8. Rerank  → Top-k results (best semantic + keyword match)
              ↓
9. Generate → LLM context from result.to_str_for_generation()
```

---

## 🔍 Retriever Comparison

### FAISS vs Qdrant

| Aspect | FAISS | Qdrant |
|--------|-------|--------|
| **Search type** | Dense only | Dense + Sparse |
| **Setup** | Local file | Local/Cloud |
| **Scaling** | Limited | Excellent |
| **Production** | Good | Better |
| **Persistence** | Manual save | Automatic |

---

## 📊 Performance

| Retriever | Speed | Accuracy | Setup |
|-----------|-------|----------|-------|
| **Qdrant (Dense)** | ~50-100ms (1084 chunks) | Semantic | Moderate |
| **Qdrant (Hybrid)** | ~100-200ms (RRF) | Semantic + Keywords | Moderate |
| **FAISS** | ~20-50ms | Semantic | Easy |

---

## 📋 Troubleshooting

| Issue | Solution |
|----------|----------|
| "Collection not found" | Call `retriever.initialize()` |
| "Wrong embedding dimension" | Ensure embedder and retriever match (384 for QdrantEmbedder) |
| "No results found" | Check query embedding and collection is not empty |
| "OOM in Qdrant" | Use FAISS or reduce batch size |
| "Slow search" | Adjust `nprobe` (FAISS) or verify Qdrant indexing |

---

## 🔧 API Reference

### QdrantRetriever

```python
# Init & setup
retriever = QdrantRetriever(config)
retriever.initialize(embedding_dimension)

# Ingestion
retriever.add_chunks(chunks, embeddings)
retriever.add_chunks_hybrid(chunks, hybrid_embeddings)

# Search
retriever.search(query_embedding, top_k=10)
retriever.search_hybrid(query_text, keywords, top_k=10)

# Persistence
retriever.save()
retriever.load()
```

### FAISSRetriever

```python
retriever = FAISSRetriever(config)
retriever.initialize(embedding_dimension)
retriever.add_chunks(chunks, embeddings)
retriever.search(query_embedding, top_k=10)
retriever.save()
retriever.load()
```

---

## 📚 Related Resources

- [Embedders Documentation](../embedders/EMBEDDERS.md) - Embedding generation
- [Chunkers Documentation](../chunkers/CHUNKERS.md) - Chunking before indexing
- [Parser Documentation](../parsers/PARSER.md) - Extraction before chunking
- [RAG Pipeline](../RAG.md) - Complete pipeline

## 🔗 Related Components

- [Query Transformers Documentation](../query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement before retrieval
- [Agentic Workflow](../../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Using retrieval results in agent decisions

---

**Built for production RAG applications** 🚀