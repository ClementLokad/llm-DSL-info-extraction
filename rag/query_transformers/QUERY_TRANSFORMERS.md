# 🔄 Query Transformers - Query Enhancement

> Transform user queries via LLM to improve semantic search and retrieval diversity.

## 📁 Folder Contents

- **`hyde_query_transformer.py`** - HyDE (Hypothetical Document Embeddings)
- **`fusion_query_transformer.py`** - Fusion (Query rewriting)
- **`query_transformer_factory.py`** - Factory pattern

---

## 🎯 Overview

| Transformer | Technique | Goal | Output |
|-------------|-----------|--------|--------|
| **HyDE** | Document generation | Generate hypothetical summaries | N technical summaries |
| **Fusion** | Query expansion | Rewrite query with synonyms | Original query + N variants |

---

## 🧠 HyDE - Hypothetical Document Embeddings

Generates multiple technical summaries answering the query.

### How it Works

```
Query: "How to calculate total revenue?"
         ↓
    LLM prompt (Envision expert mode)
         ↓
    Generate N technical summaries (~150 words each)
         ↓
Output: [
    "Items.Revenue = Items.Quantity * Items.Price; sum(Items.Revenue)",
    "Orders can join with Items table via Foreign key; aggregate revenue across orders",
    "Use cross() to link dimensions; multiply fact columns..."
]
```

### Configuration

```yaml
query_transformers:
  mode: "hyde"
  amount_of_generated_instances: "several"  # "several", "3", "5", etc.
  
agent:
  rate_limit_delay: 0.5  # Delay between LLM calls
```

### Usage

```python
from rag.query_transformers.query_transformer_factory import QueryTransformerFactory

transformer = QueryTransformerFactory.create(config)

# HyDE: Generate hypothetical documents
summaries = transformer.transform(
    query="How to calculate total revenue?",
    verbose=True
)

# Result: List of generated technical summaries
for summary in summaries:
    embedding = embedder.embed_text(summary)
    # → Search with each embedding
```

### Use Cases

- **Concept matching**: Generic query → specific technical documents
- **Semantic expansion**: Single question → multiple technical angles

---

## 🔀 Fusion - Query Expansion

Rewrites query with synonyms and alternative codebase terminology.

### How it Works

```
Query: "Calculate total revenue?"
         ↓
    LLM prompt (Query rewriting with Envision terminology)
         ↓
    Generate N variants using different terminology
         ↓
Output: [
    "Calculate total revenue?",                    # Original
    "Sum all Items.Price * Items.Quantity",        # Technical variant
    "Aggregate revenue across all transactions",   # Alternative phrasing
    "Compute the total monetary value per items"   # Synonym variant
]
```

### Configuration

```yaml
query_transformers:
  mode: "fusion"
  amount_of_generated_instances: "several"
  
agent:
  rate_limit_delay: 0.5
```

### Usage

```python
from rag.query_transformers.query_transformer_factory import QueryTransformerFactory

transformer = QueryTransformerFactory.create(config)

# Fusion: Generate query variants
variants = transformer.transform(
    query="Calculate total revenue?",
    verbose=True
)

# Result: [original_query, variant1, variant2, ...]
# All variants → embeddings → retrieve separately
```

### Use Cases

- **Terminology diversity** : Codebase uses "Total" vs "Sum" vs "Aggregate"
- **RAG Fusion** : Multiple queries → multiple retrievals → RRF fusion

---

## 🏗️ Architecture

### BaseQueryTransformer

```python
class BaseQueryTransformer(ABC):
    def __init__(self, config):
        self.agent = prepare_query_transformer_agent()
        self.generated_instances_amount = config.get('query_transformers.amount_of_generated_instances')
        self.rate_limit_delay = config.get('agent.rate_limit_delay')
    
    @abstractmethod
    def transform(self, query: str, verbose: bool = False) -> List[str]:
        """Transform query and return list of outputs"""
        pass
```

### QueryTransformerFactory

```python
transformer = QueryTransformerFactory.create(config)
# Returns HyDE or Fusion depending on config['query_transformer.query_transformer_mode']
```

---

## 📍 Real Usage

### RAG Pipeline with Transformers

```
1. User Query: "Calculate total revenue?"
         ↓
2. Transform (Fusion):
   - variant_1: "Calculate total revenue?"
   - variant_2: "Sum all Items.Price * Items.Quantity"
   - variant_3: "Aggregate monetary value per item"
         ↓
3. Embed each variant: [e1, e2, e3]
         ↓
4. Retrieve for each: 
   - e1 → [chunk1, chunk2, ...]
   - e2 → [chunk3, chunk4, ...]
   - e3 → [chunk5, chunk6, ...]
         ↓
5. Fusion (RRF): Combine results → Top-k
         ↓
6. Generate: LLM uses fused context
```

---

## 🔧 API Reference

### HyDE & Fusion (same interface)

```python
# Init
transformer = HydeQueryTransformer(config)
# or
transformer = FusionQueryTransformer(config)

# Transform
outputs = transformer.transform(query, verbose=False)
# Returns: List[str]
```

### Factory

```python
transformer = QueryTransformerFactory.create(config)
# Returns None if query_transformer_mode not set, else HyDE/Fusion instance
```

---

## 📊 Comparison

| Aspect | HyDE | Fusion |
|--------|------|--------|
| **Input** | User query | User query |
| **Output** | N technical summaries | Original + N variants |
| **LLM cost** | Higher (generate from scratch) | Medium (rewrites only) |
| **Use case** | Concept expansion | Terminology diversity |
| **Retrieved chunks** | Similar to generated summaries | Matches different keywords |

---

## 📋 Configuration

```yaml
query_transformer:
  query_transformer_mode: "fusion"  # or "hyde"
  
query_transformers:
  amount_of_generated_instances: "several"  # or "3", "5", etc.

agent:
  rate_limit_delay: 0.5  # Between LLM calls
```

---

## 📚 Related Resources

- [Embedders Documentation](../embedders/EMBEDDERS.md) - Embedding transformed queries
- [Retrievers Documentation](../retrievers/RETRIEVERS.md) - Retrieval with multiple queries
- [RAG Pipeline](../RAG.md) - Complete pipeline

## 🔗 Related Components

- [Chunkers Documentation](../chunkers/CHUNKERS.md) - Understanding chunk structure
- [Agentic Workflow](../../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Query transformation in agent loop

---

**Built for production RAG applications** 🚀