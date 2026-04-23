# 🧬 Embedders - Semantic Transformation into Vectors

> Convert code chunks into dense/sparse vector embeddings optimized for semantic search and RAG retrieval.

## 📁 Folder Contents

- **`qdrant_embedder.py`** - Hybrid embedder (dense + sparse) for Qdrant - **PRODUCTION**
- **`sentence_transformer_embedder.py`** - Local SentenceTransformer embedder
- **`gemini_embedder.py`** - Google Gemini API embedder (optional)
- **`openai_embedder.py`** - OpenAI API embedder (optional)

---

## 🎯 Embedders Overview

| Embedder | Type | Model | Features | Usage |
|----------|------|--------|------------------|-------|
| **QdrantEmbedder** | Hybrid | all-MiniLM-L6-v2 + BM25 | Dense + Sparse, no API | **Production RAG** |
| **SentenceTransformer** | Local | Configurable | Fast, no API | Alternative fallback |
| **Gemini** | API | text-embedding-004 | High quality, quota | Experimental |
| **OpenAI** | API | text-embedding-3-small | High quality, quota | Experimental |

---

## 🔌 QdrantEmbedder (Main Production)

Hybrid embedder combining semantic dense search and keyword-based sparse search via BM25.

### Hybrid Architecture

```
Chunk: "Items.Total = sum(Orders.Amount)"
  ↓
  ├─ Dense (semantic)  → all-MiniLM-L6-v2   → [384-dim vector]
  └─ Sparse (keyword)  → BM25 tokenization  → {word: score, ...}
      ↓
  Qdrant stores both for hybrid retrieval
```

### Default Models

```python
# Dense embeddings
dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension = 384

# Sparse embeddings (BM25)
sparse_model_name = "Qdrant/bm25"
disable_stemmer = True  # Keep exact keywords like "Items"
```

### Main Methods

#### 1. Semantic Embeddings (dense) - BaseEmbedder Interface

```python
from rag.embedders.qdrant_embedder import QdrantEmbedder

embedder = QdrantEmbedder(config)
embedder.initialize()

# Dense embeddings only
dense_vecs = embedder.embed_chunks(chunks)
# Shape: (N, 384)

query_embedding = embedder.embed_text(user_query)
# Shape: (384,)
```

#### 2. Hybrid Retrieval (dense + sparse) - Qdrant-specific

```python
# For insertion into Qdrant
hybrid_results = embedder.embed_chunks_hybrid(chunks)
# Result: List[{
#   "dense": [384-dim float list],
#   "sparse": SparseEmbedding{word: score}
# }]

# For queries
query_hybrid = embedder.embed_text_hybrid(
    text="How to calculate revenue?",
    keywords=["Items", "Orders", "calculate"]
)
# Result: {
#   "dense": [384-dim vector],
#   "sparse": {word: score}
# }
```

### Configuration

```yaml
embeddings:
  qdrant:
    dense_model_name: "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model_name: "Qdrant/bm25"
    embedding_dimension: 384
    disable_stemmer: true
    batch_size: 32
```

### Advantages

| Advantage | Impact |
|----------|--------|
| **No API** | Zero quota, zero network latency |
| **Hybrid** | Combines semantic + exact matching |
| **Local** | Fast, private, reproducible |
| **Versatile** | Dense for semantics, Sparse for syntax |
| **ONNX Runtime** | Very fast (~1000 chunks/sec) |

---

## 📖 SentenceTransformerEmbedder

Local embedder based on sentence-transformers (compatible with Hugging Face).

### Supported Models

```python
model_name: 'all-MiniLM-L6-v2'      # Default - 384-dim, fast
model_name: 'all-mpnet-base-v2'     # 768-dim, better quality
model_name: 'multilingual-e5-base'  # Multilingual
```

### Usage

```python
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

config = {
    "general": {
        "sentence_transformer": {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cuda",  # GPU support
            "show_progress_bar": True,
            "normalize_embeddings": True
        }
    }
}

embedder = SentenceTransformerEmbedder(config)
embedder.initialize()

# Dense embeddings
embeddings = embedder.embed_chunks(chunks)
# Shape: (N, 384)
```

### Features

- **Local** : No API calls, no quota
- **GPU-ready** : CUDA/Metal support for acceleration
- **Normalization** : Optional L2 normalization
- **Cache** : Model saved locally (`data/sentence_transformer/`)

---

## 🌐 GeminiEmbedder & OpenAIEmbedder (Optional)

Cloud API-based embedders for high semantic quality.

### Gemini

```python
model_name: "models/text-embedding-004"  # 768-dim
requests_per_minute: 1500
```

**Advantages** : High quality, generous quota  
**Disadvantages** : Network latency, limited quota, costs

### OpenAI

```python
model_name: "text-embedding-3-small"   # 1536-dim
model_name: "text-embedding-3-large"   # 3072-dim
requests_per_minute: 3000
```

**Advantages** : Best-in-class quality  
**Disadvantages** : Slower, higher costs, strict quotas

### Basic Usage (same interface)

```python
embedder = OpenAIEmbedder(config)
embedder.initialize()

embeddings = embedder.embed_chunks(chunks)  # Same as local embedders
query_emb = embedder.embed_text(query)
```

---

## 🏗️ Common Architecture

All embedders inherit from `BaseEmbedder` and implement:

```python
class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def embedding_dimension(self) -> int: ...
    
    @abstractmethod
    def initialize(self) -> None: ...
    
    @abstractmethod
    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray: ...
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...
```

### Text Preparation

All embedders apply before embedding:

```python
# For chunks: use LLM summary if available, else raw content
def prepare_chunk_for_embedding(chunk: CodeChunk) -> str:
    if chunk.metadata.get('summary'):
        return chunk.metadata['summary']
    return chunk.content

# For queries: basic cleanup
def prepare_text_for_embedding(text: str) -> str:
    return text.strip()[:512]  # Max 512 chars/tokens
```

---

## 📍 Real Usage in Project

### Index Building (all `build_*.py`)

```python
from rag.embedders.qdrant_embedder import QdrantEmbedder

embedder = QdrantEmbedder(config)
embedder.initialize()

for chunks_batch in chunk_batches:
    # Dense embeddings (standard)
    dense_vecs = embedder.embed_chunks(chunks_batch)
    
    # Hybrid (for Qdrant)
    hybrid_vecs = embedder.embed_chunks_hybrid(chunks_batch)
    
    # → Insert into vector database
```

### Complete RAG Workflow

```
1. Parsing   → 6078 blocks
2. Chunking  → 1084 chunks  
3. Embedding → Dense [1084 × 384] + Sparse BM25
              ↓
4. Indexing  → Qdrant (dense + sparse search)
              ↓
5. Query     → Query embedding (same dual format)
              ↓
6. Search    → Hybrid retrieval → Top-k chunks → LLM
```

---

## 🔍 Comparaison Hybrid vs Dense-only

### Hybrid Search (QdrantEmbedder)

```
Query: "How calculate Items.Revenue?"
  ├─ Dense: Semantic similarity [query ≈ 0.87 to chunk]
  └─ Sparse: Keyword match "Items.Revenue" → BM25 score

Result: Fusion scores → Better recall + precision
```

**Use cases** :
- Search by variable name (Items, Orders)
- Exact syntax search
- Semantic + keyword combination

### Dense-only (SentenceTransformer)

```
Query: "How calculate Items.Revenue?"
  └─ Semantic embedding → find similar chunks

Result: Pure semantics → Better for concepts
```

**Use cases** :
- Conceptual search ("profit calculation")
- Natural language queries
- Search without specific keywords

---

## 📊 Performance

| Metric | QdrantEmbedder | SentenceTransformer | OpenAI |
|--------|----------------|-------------------|--------|
| **Speed** | ~1000 chunks/sec (ONNX) | ~500 chunks/sec | ~10 chunks/sec (API) |
| **Cost** | Free (local) | Free (local) | ~$0.02 per 1M tokens |
| **Quality** | Good (384-dim) | Good (384-768-dim) | Excellent (1536-3072-dim) |
| **Setup** | Instant | Instant | Requires API key |

---

## 🔧 API Reference

### BaseEmbedder

```python
# Properties
embedder.embedding_dimension: int           # Vector size
embedder._is_initialized: bool              # Initialization state

# Methods
embedder.initialize()                       # Load model
embedder.embed_chunks(chunks) -> ndarray    # (N, D) array
embedder.embed_text(text) -> ndarray        # (D,) vector
embedder.prepare_chunk_for_embedding(chunk) # Pre-process
embedder.prepare_text_for_embedding(text)   # Pre-process
```

### QdrantEmbedder (Qdrant-specific)

```python
# Standard (from BaseEmbedder)
embedder.embed_chunks(chunks) -> ndarray
embedder.embed_text(text) -> ndarray

# Qdrant hybrid (NEW)
embedder.embed_chunks_hybrid(chunks) -> List[Dict]
embedder.embed_text_hybrid(text, keywords) -> Dict
```

---

## 📋 Troubleshooting

| Issue | Solution |
|----------|----------|
| "Model not initialized" | Call `embedder.initialize()` |
| OOM (out of memory) | Reduce `batch_size` |
| Wrong dimension | Verify `embedding_dimension` vs `dense_model_name` |
| Empty sparse embeddings | Ensure tokens exist in the text |

---

## 📚 Related Resources

### RAG Pipeline Flow
- [RAG Pipeline Overview](../RAG.md) - Complete pipeline with embedding step
- [Chunker Documentation](../chunkers/CHUNKERS.md) - Chunks before embedding
- [Retrievers Documentation](../retrievers/RETRIEVERS.md) - Using embeddings for retrieval

### Related Components
- [Parser Documentation](../parsers/PARSER.md) - Blocks before chunking
- [Query Transformers Documentation](../query_transformers/QUERY_TRANSFORMERS.md) - Embedding transformed queries
- [Agentic Workflow](../../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Embeddings in agent reasoning

### Configuration & Usage
- [Configuration](../../config.yaml) - Embedder parameters
- [Quick Start Tutorial](../../TUTORIAL.md) - Setup and usage

---

**Built for production RAG applications** 🚀
