# 📊 Benchmarks - Evaluation Metrics for RAG Agent Answers

> Production evaluation system for measuring answer quality. Multiple strategies: embedding similarity, cross-encoder ranking, LLM judgment, and hybrid deterministic+neural approaches.

---

## 🎯 Overview

**Purpose**: Evaluate how well the RAG agent answers questions about the Envision codebase.

**Key Challenge**: Code Q&A answers require different evaluation strategies:
- **Deterministic questions** ("List all scripts"): Factual accuracy check (exact match)
- **Conceptual questions** ("How does revenue calculation work?"): Semantic understanding check (meaning match)
- **Contradictory responses**: Factual consistency check (no hallucinations)

**Available Strategies** (5 total):
| Strategy | Method | Cost | Use Case |
|----------|--------|------|----------|
| **CosineSimBenchmark** | Embedding similarity | Low | Fast semantic comparison |
| **DualBenchmark** | Cross-encoder NLI + relevance | Medium | Balanced quality assessment |
| **LLMAsAJudgeBenchmark** | LLM judge (binary 0/1) | High | Strict factual grading |
| **LLMAsAJudgeBenchmark2** | LLM judge (1-5 scale) | High | Nuanced grading |
| **HybridBenchmark** | Deterministic check + DualBenchmark | Medium-High | Best of both worlds |

---

## 🏗️ Architecture

### **Base Class** (`base_benchmark.py`)

```python
class Benchmark:
    """Base class for all benchmarks."""
    
    def initialize(self):
        """Optional setup (load models, init connections)."""
        pass
    
    def run(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Args:
            data: List of evaluation items
                [
                    {
                        "question": str,
                        "llm_response": str,
                        "reference": str,
                        "deterministic": bool (optional),
                    },
                    ...
                ]
        
        Returns:
            {
                "results": [
                    {"question": ..., "score": ..., ...},
                    ...
                ],
                "mean_score": float
            }
        """
        raise NotImplementedError()
```

**Interface Contract**:
- All benchmarks implement `run(data) → Dict[str, Any]`
- Returns standardized format: `{"results": [...], "mean_score": float}`
- Supports pluggable evaluation strategies

---

## 📏 Evaluation Strategies

### **1. CosineSimBenchmark** - Semantic Similarity

**Purpose**: Measure embedding-space distance between LLM response and reference.

**Method**:
1. Embed LLM response using SentenceTransformer
2. Embed reference answer using same model
3. Compute cosine similarity: `cos_sim(emb1, emb2) ∈ [0, 1]`

**Implementation Choice: SentenceTransformer**
- ✅ Fast (CPU inference, ~10-50ms per pair)
- ✅ Deterministic (same model state = same embeddings)
- ✅ Generalizable (trained on semantic tasks)
- ❌ May miss fine-grained code syntax
- ❌ Penalizes correct but differently-phrased answers

**Usage**:
```python
benchmark = CosineSimBenchmark()
# (embedder auto-initialized from config)

results = benchmark.run([
    {
        "question": "How to calculate revenue?",
        "llm_response": "Use Quantity * Price",
        "reference": "Multiply quantity by price",
    }
])
# {"results": [{"score": 0.87, ...}], "mean_score": 0.87}
```

**Score Interpretation**:
- 0.9+: Very similar meaning
- 0.7-0.9: Similar but with some rephrasing
- 0.5-0.7: Related concepts, partial overlap
- <0.5: Dissimilar or off-topic

**Best For**:
- Quick baseline evaluation
- Conceptual questions (e.g., "Explain the flow")
- When code syntax isn't critical
- Budget-conscious benchmarking

---

### **2. DualBenchmark** - Cross-Encoder Ranking

**Purpose**: Balance factual entailment + query relevance using specialized neural models.

**Architecture**:
```
Step 1: Entailment Check (NLI)
  reference answer + generated answer
  → Cross-Encoder (nli-deberta-v3-base)
  → Outputs: [contradiction_prob, entailment_prob, neutral_prob]

Step 2: Query Relevance Check
  query + generated answer
  → Cross-Encoder (jinaai/jina-reranker-v2-base-multilingual)
  → Outputs: relevance logit → sigmoid → relevance_prob

Step 3: Composite Score
  IF contradiction_prob > 0.5:
    score = 0.0  (hallucination detected)
  ELSE:
    score = (entailment_prob × 0.6) + (relevance_prob × 0.4)
    # 60% strict fact match + 40% overall helpfulness
```

**Why Dual Models?**
- NLI (Natural Language Inference): Detects contradictions ("X implies NOT X")
- Reranker: Measures query-response alignment (avoids off-topic answers)
- Composite: Balances facts + usefulness

**Usage**:
```python
benchmark = DualBenchmark()
benchmark.initialize()  # Loads large cross-encoder models (~500MB)

results = benchmark.run([
    {
        "question": "Which scripts import items?",
        "llm_response": "The files orders.nvn and forecast.nvn...",
        "reference": "orders.nvn and forecast.nvn import items",
    }
])
# score = 0.6×1.0 (entailed) + 0.4×0.95 (relevant) = 0.98
```

**Score Breakdown** (in returned results):
```python
{
    "score": 0.78,                 # Composite (0-1)
    "entailment": 0.95,            # Fact match (0-1)
    "relevance": 0.92,             # Query relevance (0-1)
    "contradiction": 0.05,         # Hallucination risk (0-1)
}
```

**Best For**:
- Balanced quality assessment
- Detecting hallucinations + relevance
- Medium token budget
- Production-grade evaluation

---

### **3. LLMAsAJudgeBenchmark** - LLM Judge (Binary 0/1)

**Purpose**: Strict LLM-based grading: answer is either correct or not.

**Method**:
1. Prompt an LLM with reference + generated answer
2. LLM evaluates: "Does response contain all key facts from reference?"
3. Returns binary score: 0 (missing info/wrong) or 1 (complete/correct)

**System Prompt**:
```
Tu es un évaluateur strict.

CRITÈRES:
- Si la réponse du LLM est très incomplète ou contient des hallucinations → 0
- Si la réponse du LLM contient tous les éléments clés → 1

FORMAT REQUIS:
JSON avec "reasoning" (justification) et "score" (0 ou 1)
```

**Usage**:
```python
benchmark = LLMAsAJudgeBenchmark()
benchmark.initialize()

results = benchmark.run([
    {
        "question": "List all scripts that read from Items table",
        "llm_response": "orders.nvn, forecast.nvn",
        "reference": "orders.nvn, forecast.nvn, analysis.nvn"
    }
])
# score = 0 (missing analysis.nvn)
```

**Returned Results**:
```python
{
    "score": 0,                    # Binary 0 or 1
    "reasoning": "La réponse omet analysis.nvn"
}
```

**Why LLM Judge?**
- ✅ Flexible criteria (can understand context)
- ✅ Catches hallucinations (LLM can identify false claims)
- ✅ Works for any question type
- ❌ High token cost (1 LLM call per question)
- ❌ Non-deterministic (different runs may vary)
- ❌ Can make mistakes

**Best For**:
- Strict factual evaluation
- Small benchmark sets (<50 questions)
- When accuracy matters more than speed
- Validating deterministic answers

---

### **4. LLMAsAJudgeBenchmark2** - LLM Judge (1-5 Scale)

**Purpose**: Nuanced LLM grading with finer-grained scale.

**Grading Scale**:
- **5**: Perfect. Absolute accuracy (exact numbers/lists, or concept perfectly explained)
- **4**: Very good. Slight omission or minor detail missing, but essentials correct
- **3**: Partial. Contains correct elements but has major omissions or minor hallucinations
- **2**: Bad. Strong hallucinations or off-topic
- **1**: Wrong. Totally incorrect or contradicts reference

**System Prompt Strategy**:
```
Question type detection:
- TECHNICAL (e.g., "Count scripts", "List files")
  → Be harsh. Missing one item = major penalty
  → Need exact numbers/lists
  
- CONCEPTUAL (e.g., "How does X work?")
  → More lenient. Verbose but correct = OK
  → Evaluate semantic correctness not verbosity
```

**Usage**:
```python
benchmark = LLMAsAJudgeBenchmark2()
benchmark.initialize()

results = benchmark.run([
    {
        "question": "How does revenue calculation work?",
        "llm_response": "Revenue is computed by multiplying quantity...",
        "reference": "Items.Revenue = Items.Quantity * Items.Price"
    }
])
# score could be 4 (correct concept, verbose)
```

**Returned Results**:
```python
{
    "score": 4,                    # 1-5 scale, normalized to 0.0-1.0
    "reasoning": "Explication complète mais plus verbeuse que nécessaire"
}
```

**Score Normalization**:
- Internal: 1-5
- Returned in `mean_score`: 0.0-1.0 (divided by 5)
- Allows comparison with other benchmarks

**Best For**:
- Mixed question types (technical + conceptual)
- Nuanced evaluation requirements
- When some partial credit is acceptable
- Larger benchmarks (100+ questions)

---

### **5. HybridBenchmark** - Best of Both Worlds

**Purpose**: Deterministic check for factual questions, neural ranking for conceptual.

**Architecture**:
```
FOR EACH QUESTION:
  IF question.deterministic == True:
    # Questions like "List 3 scripts"
    ├─ Split reference into lines
    ├─ Normalize both (lowercase, remove punctuation, collapse whitespace)
    ├─ Check each reference line appears in response
    └─ Score = 1.0 if ALL matched, else 0.0
    
  ELSE:
    # Questions like "How does X work?"
    └─ Use DualBenchmark (neural scoring)
```

**Implementation Choice: Hybrid**
- ✅ Factual questions get binary grading (simple, fair)
- ✅ Conceptual questions get balanced scoring (nuanced)
- ✅ Avoids LLM judge overhead while being fair
- ✅ Deterministic results for reproducibility

**Usage**:
```python
benchmark = HybridBenchmark(underministic_benchmark="dual")
benchmark.initialize()

results = benchmark.run([
    {
        "question": "Which scripts import Items?",
        "llm_response": "orders.nvn and forecast.nvn",
        "reference": ["orders.nvn", "forecast.nvn"],
        "deterministic": True  # ← Key flag
    },
    {
        "question": "Explain revenue calculation",
        "llm_response": "Revenue is quantity times price...",
        "reference": "Multiply quantity by price",
        "deterministic": False  # ← Use DualBenchmark
    }
])
```

**Returned Results**:
```python
{
    "results": [
        {
            "score": 1.0,
            "method": "deterministic_check",
            "matched": "2/2",
            "normalized_response": "ordersnnvforecastnnv",
            "normalized_reference_lines": ["ordersnnv", "forecastnnv"]
        },
        {
            "score": 0.87,
            "method": "dual_benchmark",
            "entailment": 0.95,
            "relevance": 0.92,
            ...
        }
    ],
    "mean_score": 0.935
}
```

**Best For**:
- Production evaluation (combines strengths)
- Mixed question datasets
- Code Q&A systems (facts + concepts)
- Maximum accuracy with reasonable cost

---

## 🔄 Comparison & Selection Guide

```
┌─────────────────────┬──────────┬────────┬──────────┬─────────────┐
│ Benchmark           │ Speed    │ Cost   │ Accuracy │ Best Use    │
├─────────────────────┼──────────┼────────┼──────────┼─────────────┤
│ CosineSimBenchmark  │ ⚡ Fast │ 💰 $   │ ★★☆     │ Baseline    │
│ DualBenchmark       │ 🚗 Med  │ 💰💰$  │ ★★★★   │ Balanced    │
│ LLMAsAJudge (0/1)   │ 🚂 Slow │ 💰💰💰| ★★★★★  │ Strict      │
│ LLMAsAJudge (1-5)   │ 🚂 Slow │ 💰💰💰| ★★★★★  │ Nuanced     │
│ HybridBenchmark     │ 🚗 Med  │ 💰💰$ │ ★★★★★  │ Production  │
└─────────────────────┴──────────┴────────┴──────────┴─────────────┘
```

**Decision Tree**:

```
Question Type?
│
├─ FACTUAL ("List X", "Count Y")
│  └─ HybridBenchmark (deterministic=True)
│      └─ Exact match scoring
│
├─ CONCEPTUAL ("How does", "Explain")
│  └─ HybridBenchmark (deterministic=False)
│      └─ DualBenchmark (semantic + relevance)
│
└─ BUDGET CONSTRAINED?
   ├─ YES → CosineSimBenchmark (fastest, cheap)
   └─ NO → HybridBenchmark (best all-around)
```

---

## 📊 Evaluation Data Format

### **Input Format**

```python
data = [
    {
        "question": str,           # The question asked
        "llm_response": str,       # Agent's answer
        "reference": str | list,   # Ground truth
        "deterministic": bool,     # (Optional) For HybridBenchmark
    },
    ...
]
```

### **Output Format**

```python
{
    "results": [
        {
            "question": str,
            "llm_response": str,
            "reference": str,
            "score": float,        # 0.0-1.0
            "method": str,         # Which benchmark strategy
            # Strategy-specific fields:
            "reasoning": str,      # (LLM judge)
            "entailment": float,   # (DualBenchmark)
            "relevance": float,    # (DualBenchmark)
            "contradiction": float,# (DualBenchmark)
            "matched": str,        # (HybridBenchmark)
        },
        ...
    ],
    "mean_score": float,          # Average score across all items
    "issues": int,                # (Optional) Evaluation errors
}
```

---

## 🎬 Running Benchmarks

### **Simple Example**

```python
from pipeline.benchmarks.hybrid_benchmark import HybridBenchmark

# Initialize
benchmark = HybridBenchmark(underministic_benchmark="dual")
benchmark.initialize()

# Prepare data
questions = [
    {
        "question": "List all scripts that import Items",
        "llm_response": "orders.nvn, forecast.nvn, analysis.nvn",
        "reference": ["orders.nvn", "forecast.nvn", "analysis.nvn"],
        "deterministic": True
    },
    {
        "question": "How does the Items table work?",
        "llm_response": "The Items table stores product inventory...",
        "reference": "Items stores quantity and price information",
        "deterministic": False
    }
]

# Run
results = benchmark.run(questions)

# Display
print(f"Mean Score: {results['mean_score']:.2%}")
for r in results['results']:
    print(f"  {r['question']}: {r['score']:.2%}")
```

### **Full Benchmark Suite**

```bash
# Run complete benchmark from CLI
python -m pipeline.benchmarks.benchmark_for_benchmark

# Output: Compares judge predictions against golden labels
```

---

## 📈 Statistics Collection

**Integrated with AgenticPipeline** (via `stats_collector.py`):

```python
collector = get_collector()

# During benchmark execution:
collector.start_benchmark()

# Track tool usage
collector.record_tool_call("rag_tool")
collector.start_tool_execution("rag_tool")
# ... tool runs ...
collector.end_tool_execution("rag_tool")

# Track LLM generation
collector.start_llm_generation("solver")
# ... LLM call ...
collector.end_llm_generation("solver")

# Track rate limiting
collector.record_rate_limit_delay(0.5)

collector.end_benchmark()

# Export statistics
stats = collector.get_stats()
```

**Collected Metrics**:
```python
{
    "tool_call_counts": {
        "rag_tool": 3,
        "grep_tool": 1,
        "graph_tool": 0,
        ...
    },
    "llm_generation_times": {
        "planner-initial-call": 2.3,
        "planner-followup-call": 1.8,
        "solver": 4.2,
        "cleaning": 0.9,
        "distillation": 3.1,
        "grader": 5.0,
    },
    "tool_execution_times": {
        "rag_tool": 0.45,
        "grep_tool": 0.12,
        ...
    },
    "rate_limit_delay_total": 1.5,
    "total_benchmark_time": 23.4,
}
```

---

## 🔍 Debugging & Visualization

### **View Benchmark Results**

```bash
# Display results with rich formatting
python view_benchmark_results.py results.json

# Show only summary (no details)
python view_benchmark_results.py results.json --quiet

# Display model + token info
python view_benchmark_results.py results.json --models --tokens
```

### **Export Format**

Benchmarks export JSON with:
```python
{
    "Results": [
        {
            "id": "Q1",
            "question": "...",
            "llm_response": "...",
            "reference": "...",
            "score": 0.95,
            ...
        }
    ],
    "timestamp": "2026-04-23T14:30:00",
    "benchmark_type": "HybridBenchmark",
}
```

---

## 📋 Configuration

```yaml
# config.yaml
main_pipeline:
  # Benchmark selection
  benchmark_type: "hybrid"  # "cosine", "dual", "llm", "llm2", "hybrid"
  
  # For HybridBenchmark
  hybrid_benchmark:
    underministic_benchmark: "dual"  # "dual" or "llm"
    normalize_scores: true
    
# Embedding config (for CosineSimBenchmark)
embedding:
  type: "sentence-transformer"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"  # or "cpu"
```

---

## 🚀 Production Guidelines

### **Choosing a Benchmark**

| Scenario | Recommendation | Rationale |
|----------|---|---|
| **Initial prototyping** | CosineSimBenchmark | Fast feedback loop |
| **Quality assurance** | HybridBenchmark | Covers factual + conceptual |
| **Regression testing** | HybridBenchmark | Reproducible deterministic check |
| **Production validation** | HybridBenchmark + DualBenchmark | Two-stage approval |
| **Research/Analysis** | All 5 (compare strategies) | Study benchmark behavior |

### **Best Practices**

1. **Always tag questions**: Mark deterministic vs conceptual
   ```python
   "deterministic": True  # for "List X" questions
   "deterministic": False # for "Explain Y" questions
   ```

2. **Verify ground truth**: Reference answers must be accurate
   - Manual review by domain expert
   - Cross-check against actual codebase

3. **Use HybridBenchmark by default**: Best balance of accuracy + cost

4. **Monitor individual scores**: `results[i]['score']` to identify weak areas

5. **Track over time**: Compare mean_score across benchmark runs
   - Improvement: System is working better
   - Degradation: Something broke or changed

---

## 📚 Related Documentation

- [Agentic Workflow](../agent_workflow/AGENTIC_WORKFLOW.md) - Agent being evaluated
- [RAG Pipeline](../../rag/RAG.md) - Retrieval system components
- [Stats Reporter](../stats_reporter.py) - Detailed benchmark metrics
- [Answer Validation](../answer_validation.py) - Path verification

---

**Production-grade evaluation system for code Q&A quality assurance** ✅