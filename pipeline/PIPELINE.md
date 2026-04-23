# 🔄 Pipeline - LangGraph Orchestration for Agentic RAG

> Production-grade LangGraph pipeline managing both single Q/A queries and benchmark evaluation of entire test suites. Orchestrates agentic workflow, answer validation, grading, and statistics collection.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DSL Query System (main.py)                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
         ┌──────────────┐         ┌──────────────────┐
         │ Single Q/A   │         │ Benchmark        │
         │ (Query)      │         │ (Multiple Q/A)   │
         └──────┬───────┘         └────────┬─────────┘
                │                          │
                ▼                          ▼
        ┌────────────────┐      ┌───────────────────────┐
        │ LangGraph      │      │ Benchmark LangGraph   │
        │ Single QA      │      │ Multi-Question Loop   │
        │ Pipeline       │      │                       │
        │                │      │ ┌─────────────────┐   │
        │ ┌────────────┐ │      │ │ For each Q:     │   │
        │ │  Agentic   │ │      │ │ ├─ Invokes      │   │
        │ │  Workflow  │◄─────────┤ │  Single QA    │   │
        │ │  (SubGraph)│ │      │ │ ├─ Collects     │   │
        │ │            │ │      │ │ │  results      │   │
        │ └────────────┘ │      │ │ └─ Loops        │   │
        │                │      │ └─────────────────┘   │
        │ ┌────────────┐ │      │                       │
        │ │  Solver    │ │      │ ┌─────────────────┐   │
        │ │  LLM       │ │      │ │ Aggregate &     │   │
        │ └────────────┘ │      │ │ Grade Results   │   │
        │                │      │ └─────────────────┘   │
        │ ┌────────────┐ │      │                       │
        │ │  Answer    │ │      │ ┌─────────────────┐   │
        │ │  Validator │ │      │ │ Export Stats    │   │
        │ └────────────┘ │      │ │ (JSON, CSV)     │   │
        │                │      │ └─────────────────┘   │
        │ ┌────────────┐ │      │                       │
        │ │  Grader    │ │      │                       │
        │ │ (Benchmark)│ │      │                       │
        │ └────────────┘ │      │                       │
        └────────────────┘      └───────────────────────┘
                │                          │
                └──────────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Statistics Collector │
                    │ - Tool calls         │
                    │ - LLM timings        │
                    │ - Token counts       │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Results & Metrics    │
                    │ Displayed + Exported │
                    └──────────────────────┘
```

---

## 🗂️ Pipeline Directory Structure

### **1. Core Files**

#### **`langgraph_base.py`** - State & Base Classes

**Defines**:
- State schemas (TypedDict for LangGraph)
- Base classes for pipeline nodes

**Key States**:

```python
# Basic RAG query state
GraphState(TypedDict):
    question: str              # User's question
    reference_answer: str      # Ground truth (benchmarking)
    retrieved_context: List[]  # Retrieved chunks
    prompt: str                # Final prompt to LLM
    generation: str            # Raw LLM output
    final_answer: str          # Validated answer
    regenerate_needed: bool    # Loop control
    retry_count: int           # Prevent infinite loops
    grade: Dict                # Benchmark score
    verbose: bool              # Debug output
    deterministic: bool        # Question type flag

# Advanced agent state (with knowledge bank + history)
AgentGraphState(GraphState):
    knowledge_bank: List[KnowledgeElement]      # Extracted facts
    execution_history: List[ActionLog]          # Tool call history
    accumulated_evidence: Dict[str, Result]     # Evidence cache (for reuse)
    undistilled_log: ActionLog                  # Raw results not yet distilled
    answer_validation_report: Dict              # Path validation feedback

# Benchmark state (multiple Q/A)
BenchmarkState(TypedDict):
    qa_pairs: List[Tuple[str, str]]             # List of (question, reference)
    qa_metadata: Dict[str, Dict]                # Question → {deterministic, ...}
    grades: List[Dict]                          # Grading results per Q
    benchmark_results: Dict                     # Aggregated stats
    sub_rag_system: StateGraph                  # Single Q/A subgraph
    verbose: bool
```

**Base Classes**:
- `BasePipeline`: Abstract nodes for retrieval, validation, grading
- `ActionLog`: Represents a tool call step
- `KnowledgeElement`: Represents extracted facts with evidence tracking

---

#### **`answer_validation.py`** - Path Verification

**Purpose**: Validate file paths cited in final answers against actual codebase.

**Key Components**:

```python
class SourcePathValidator:
    """Lightweight validator for script paths in answers."""
    
    def extract_candidates(answer: str) -> List[str]:
        """Extract potential file paths from answer text.
        
        Uses 4 regex patterns:
        1. Backtick-enclosed paths: `path/to/file.nvn`
        2. Full paths with extensions: /path/to/file.nvn
        3. Numbered folder structures: /2/subfolder/file
        4. Simple .nvn files: script.nvn
        """
        
    def validate_path(candidate: str) -> Optional[str]:
        """Check if path exists in codebase (with fuzzy matching).
        
        Features:
        - ignore_extension: treats .nvn ≈ .nvm
        - ignore_leading_slash: /path ≈ path
        - allow_partial_suffix_match: path/script ≈ full/path/script
        - ignore_data_extensions: ignores .ion, .csv in comparison
        
        Returns: Canonical path if valid, None otherwise
        """
```

**Validation Strategy**:
```
answer: "See /3/Forecasting/forecast.nvn for details"
    ↓ extract_candidates()
    ["/3/Forecasting/forecast.nvn"]
    ↓ normalize_candidate_path()
    ["3/forecasting/forecast"]  (lowercased, no extension, no slash)
    ↓ validate_path()
    Match against canonical_paths from codebase
    ✓ Valid → Include in report
    ✗ Invalid → Flag as unverified
```

**Output**:
```python
{
    "validation_report": {
        "valid_paths": ["/3/Forecasting/forecast.nvn"],
        "invalid_paths": [],
        "unverified_paths": [],
        "errors": []
    },
    "all_paths_valid": True
}
```

---

#### **`stats_collector.py`** - Metrics Collection

**Purpose**: Collect timing and tool usage statistics during benchmark execution.

**Metrics Tracked**:

```python
class BenchmarkStatsCollector:
    def __init__(self):
        # Tool call frequency
        self.tool_call_counts: Dict[str, int]  
        # {"rag_tool": 12, "grep_tool": 3, "graph_tool": 1, ...}
        
        # LLM generation time per role
        self.llm_generation_times: Dict[str, float]
        # {
        #   "planner-initial-call": 2.3,
        #   "planner-followup-call": 1.8,
        #   "solver": 4.2,
        #   "cleaning": 0.9,
        #   "distillation": 3.1,
        #   "grader": 5.0,
        # }
        
        # Tool execution time
        self.tool_execution_times: Dict[str, float]
        # {"rag_tool": 0.45, "grep_tool": 0.12, ...}
        
        # Rate limiting overhead
        self.rate_limit_delay_total: float
        # Total time spent sleeping for API limits
```

**API**:

```python
collector = get_collector()  # Singleton

# Start/end benchmark
collector.start_benchmark()
collector.end_benchmark()

# Tool call tracking
collector.record_tool_call("rag_tool")
collector.start_tool_execution("rag_tool")
# ... tool runs ...
collector.end_tool_execution("rag_tool")

# LLM generation tracking
collector.start_llm_generation("solver")
# ... LLM call ...
collector.end_llm_generation("solver")

# Rate limit overhead
collector.record_rate_limit_delay(0.5)

# Get report
report = collector.get_report()
```

**Report Structure**:
```python
{
    "tool_call_counts": {...},
    "llm_generation_times": {...},
    "tool_execution_times": {...},
    "rate_limit_delay_total": 1.5,
    "total_benchmark_time": 23.4,
    "timestamp": "2026-04-23T14:30:00"
}
```

---

#### **`stats_reporter.py`** - Result Formatting

**Purpose**: Convert statistics into human-readable reports and export to files.

**Functions**:

```python
def format_stats_report(
    stats: BenchmarkStatsCollector,
    include_tokens: bool = False,
    token_stats: Optional[dict] = None
) -> str:
    """
    Generate Markdown report with:
    - Total benchmark time (highlighted)
    - Tool call counts (table)
    - LLM generation times per role (table with % of total)
    - Tool execution times (table with % of total)
    - Rate limit delay (with % of total)
    - Token usage (optional)
    
    Returns: Formatted Markdown string
    """

def save_stats_to_json(
    stats: BenchmarkStatsCollector,
    output_path: str,
    benchmark_type: str,
    token_stats: Optional[dict] = None
) -> None:
    """
    Export stats to JSON file with:
    - Timestamp
    - Benchmark type
    - All metrics
    - Token counts (if available)
    
    Output: data/benchmark_results/stats_TIMESTAMP.json
    """
```

**Example Report Output**:
```
# Benchmark Statistics

## ⏱️ Total Benchmark Time: **23.40s**

## 🔧 Tool Call Counts

| Tool | Count |
|------|-------|
| rag_tool | 12 |
| grep_tool | 3 |
| graph_tool | 1 |

## 🤖 LLM Generation Times

| Role | Duration | % of Total |
|------|----------|-----------|
| solver | 4.20s | 17.9% |
| distillation | 3.10s | 13.2% |
| planner-initial-call | 2.30s | 9.8% |

## ⏳ Rate Limit Delay

**1.50s** (6.4% of total time)
```

---

### **2. Sub-directories**

#### **`agent_workflow/`** - Agentic Decision-Making

**Documentation**: [agent_workflow/AGENTIC_WORKFLOW.md](agent_workflow/AGENTIC_WORKFLOW.md)

**Key Components**:
- `agentic_pipeline.py` - Orchestrates agentic sub-graph + solver + cleaner
- `concrete_workflow.py` - Strategic planner with Mistral tool-calling
- `workflow_base.py` - Base classes for tools (RAG, Grep, Graph, etc.)
- **7 Tools**: RAG, Grep, Graph, Script Finder, Tree, Prior Evidence, Distillation

**Role in Pipeline**:
```
Question
    ↓
[AGENTIC WORKFLOW] (Subgraph)
├─ Planner LLM (strategic decisions)
├─ Tool Loop (7 tools available)
├─ Distillation (extract facts)
├─ Solver LLM (generate answer)
└─ Cleaner LLM (format output)
    ↓
Answer
```

---

#### **`benchmarks/`** - Evaluation Metrics

**Documentation**: [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)

**Key Components**:
- `base_benchmark.py` - Abstract interface
- `cosine_sim_benchmark.py` - Embedding similarity
- `dual_cross_encoder_benchmark.py` - Cross-encoder ranking
- `llm_as_a_judge_benchmark.py` - LLM judge (binary 0/1)
- `llm_as_a_judge_benchmark.py` - LLM judge (1-5 scale)
- `hybrid_benchmark.py` - Deterministic + neural

**Role in Pipeline**:
```
Final Answer + Reference
    ↓
[BENCHMARK SELECTION] (config)
├─ CosineSimBenchmark (fast)
├─ DualBenchmark (balanced)
├─ LLMAsAJudge (strict)
└─ HybridBenchmark (production)
    ↓
Score (0.0-1.0)
```

---

## 🔄 Pipeline Phases

### **Phase 1: Single Q/A Query**

**Entry Point**: `DSLQuerySystem.query(question, verbose=True)`

**Graph Nodes** (in order):

```
1. run_agentic_workflow
   Input: question + context
   Output: knowledge_bank, execution_history, answer candidate
   Action: Invoke ConcreteAgentWorkflow sub-graph

2. check_agent_logic
   Input: answer candidate, execution history
   Decision: Does answer need more refinement?
   Output: regenerate_needed = True/False

3. decide_after_logic_check [DECISION NODE]
   Routes to either:
   ├─ "regenerate" → back to run_agentic_workflow (loop)
   └─ "clean" → continue to cleaning

4. clean_generated_answer
   Input: raw answer + system prompt
   Output: formatted, final answer
   Action: Invoke Cleaner LLM

5. validate_paths
   Input: final answer
   Output: validation report
   Action: Extract and verify file paths

6. grade_answer
   Input: final answer + reference
   Output: score
   Action: Run configured benchmark

7. END
   Output: final_answer, grade, validation_report
```

**State Flow**:
```python
Initial State:
{
    "question": "How to calculate revenue?",
    "reference_answer": "Items.Revenue = Items.Quantity * Items.Price",
    "verbose": True,
    "deterministic": False
}

After run_agentic_workflow:
{
    ...
    "knowledge_bank": [KnowledgeElement(...), ...],
    "execution_history": [ActionLog(...), ...],
    "generation": "Revenue is calculated by..."
}

After clean_generated_answer:
{
    ...
    "final_answer": "Revenue = Quantity × Price. (items.nvn:45-50)"
}

After validate_paths:
{
    ...
    "answer_validation_report": {
        "valid_paths": ["items.nvn"],
        "all_paths_valid": True
    }
}

Final State:
{
    ...
    "grade": {
        "score": 0.92,
        "entailment": 0.95,
        ...
    }
}
```

---

### **Phase 2: Benchmark (Multiple Q/A)**

**Entry Point**: `DSLQuerySystem.benchmark(qa_pairs, verbose=True)`

**Two-Level Architecture**:

```
Outer Loop (LangGraph - manages multiple questions):
├─ For each Q in qa_pairs:
│  ├─ Invoke single_qa_graph (inner LangGraph)
│  │  ├─ run_agentic_workflow
│  │  ├─ check_agent_logic
│  │  ├─ clean_generated_answer
│  │  ├─ validate_paths
│  │  └─ grade_answer
│  │      ↓ Returns score
│  ├─ Collect grade
│  ├─ Track stats
│  └─ Loop to next Q
│
└─ After all Q:
   ├─ Aggregate scores (mean, distribution)
   ├─ Compile statistics (tool counts, timing)
   ├─ Generate reports (Markdown, JSON)
   └─ Export results
```

**Graph Nodes**:

```
1. initialize_benchmark
   Input: qa_pairs list
   Output: Initialize state, load benchmarker
   
2. loop_process_questions
   Input: remaining qa_pairs
   Decision: More questions?
   Routes to either:
   ├─ "continue" → process_next_question
   └─ "done" → finalize_benchmark

3. process_next_question
   Input: current qa_pair
   Action: Invoke single_qa_graph.invoke()
   Output: grade for this Q
   
4. aggregate_grades
   Input: all grades
   Output: mean_score, statistics
   
5. compile_benchmark_report
   Input: grades + stats
   Output: formatted report
   
6. finalize_benchmark
   Input: benchmark report
   Output: save to disk, display results
```

**State Flow**:
```python
Initial:
{
    "qa_pairs": [
        ("Q1", "ref1"),
        ("Q2", "ref2"),
        ("Q3", "ref3"),
    ],
    "qa_metadata": {
        "Q1": {"deterministic": True},
        "Q2": {"deterministic": False},
        ...
    },
    "grades": [],
    "benchmark_results": None
}

After loop (processing all Q):
{
    ...
    "grades": [
        {"question": "Q1", "score": 1.0, ...},
        {"question": "Q2", "score": 0.87, ...},
        {"question": "Q3", "score": 0.92, ...},
    ]
}

After aggregation:
{
    ...
    "benchmark_results": {
        "mean_score": 0.93,
        "scores": [1.0, 0.87, 0.92],
        "min_score": 0.87,
        "max_score": 1.0,
        "std_dev": 0.06
    }
}

Final Output (saved + displayed):
{
    "timestamp": "2026-04-23T14:30:00",
    "benchmark_type": "HybridBenchmark",
    "qa_count": 3,
    "mean_score": 0.93,
    "tool_call_counts": {...},
    "llm_generation_times": {...},
    ...
}
```

---

## 🚀 Entry Points

### **Single Query**

```python
from main import MainAgenticPipeline, DSLQuerySystem
from rich.console import Console

# Initialize
console = Console()
pipeline = MainAgenticPipeline(console)
system = DSLQuerySystem(pipeline, console)

# Query
answer = system.query("How to calculate revenue?", verbose=True)
print(answer)
```

### **Benchmark Suite**

```python
# Load questions
qa_pairs = [
    ("Q1", "Expected answer 1"),
    ("Q2", "Expected answer 2"),
    ("Q3", "Expected answer 3"),
]

# Run benchmark
results = system.benchmark(qa_pairs, verbose=True)

# Results are saved to:
# data/benchmark_results/questions_TIMESTAMP.json
# data/benchmark_results/stats_TIMESTAMP.json
```

### **Command-line**

```bash
# Interactive query mode
python main.py

# Benchmark mode
python main.py --benchmark questions.json

# With custom config
python main.py --config custom_config.yaml --benchmark questions.json
```

---

## 📋 Configuration

```yaml
# config.yaml
main_pipeline:
  # LLM selection
  agent_logic:
    planner_llm: "mistral"
    main_llm: "claude"
    cleaning_llm: "claude"
    distillation_llm: "claude"
    max_retries: 2
  
  # Benchmark selection
  benchmark_type: "hybrid"  # "cosine", "dual", "llm", "llm2", "hybrid"
  
  # Answer validation
  answer_validation:
    ignore_extension: true
    allow_partial_suffix_match: true
    ignore_leading_slash: true
    ignore_data_extensions: true
    ignored_path_extensions: ["ion", "csv"]

# Tool configuration
main_pipeline:
  rag_tool:
    advanced: true          # Use AdvancedRAGTool vs SimpleRAGTool
  graph_tool:
    auto_build: true

# Rate limiting
agent:
  rate_limit_delay: 0.5    # Delay between API calls (seconds)
```

---

## 🔗 Integration Map

```
main.py (entry point)
    ↓
MainAgenticPipeline
    ├─ Initialize tools
    │  ├─ RAG (embedder + retriever + query_transformer)
    │  ├─ Grep (pattern matching)
    │  ├─ Graph (dependency queries)
    │  ├─ Script Finder (file reading)
    │  ├─ Tree (structure display)
    │  ├─ Distillation (fact extraction)
    │  └─ Prior Evidence (cache)
    │
    ├─ Select benchmark
    │  └─ Configure HybridBenchmark (default)
    │
    └─ Pre-compile ConcreteAgentWorkflow (subgraph)
        └─ Planner + 7 Tools

DSLQuerySystem.query()
    ├─ Build single_qa_graph from pipeline
    ├─ Compile to executable app
    ├─ Invoke with question
    ├─ Collect stats
    └─ Return answer

DSLQuerySystem.benchmark()
    ├─ Load qa_pairs
    ├─ Initialize benchmark_graph
    ├─ For each Q:
    │  ├─ Invoke single_qa_graph
    │  ├─ Collect grade
    │  └─ Update stats
    ├─ Aggregate results
    ├─ Export to JSON
    └─ Display report
```

---

## 📊 Complete Data Flow

```
USER INPUT QUESTION
    ↓
┌──────────────────────────────────────┐
│ AGENTIC WORKFLOW (subgraph)          │
├──────────────────────────────────────┤
│ Planner selects tools                │
│ Tools: RAG/Grep/Graph/Script/Tree   │
│ Distillation extracts facts          │
│ Solver generates answer              │
│ Cleaner formats output               │
└──────────────────────────────────────┘
    ↓ final_answer
┌──────────────────────────────────────┐
│ PATH VALIDATION                      │
├──────────────────────────────────────┤
│ Extract file paths from answer       │
│ Verify against codebase              │
│ Generate validation report           │
└──────────────────────────────────────┘
    ↓ final_answer + validation_report
┌──────────────────────────────────────┐
│ BENCHMARK EVALUATION                 │
├──────────────────────────────────────┤
│ Compare with reference answer        │
│ Generate score (0.0-1.0)             │
│ Track reasoning (why this score?)    │
└──────────────────────────────────────┘
    ↓ score + grade details
┌──────────────────────────────────────┐
│ STATISTICS COLLECTION                │
├──────────────────────────────────────┤
│ Tool call counts                     │
│ LLM generation times                 │
│ Tool execution times                 │
│ Rate limit delays                    │
│ Total benchmark time                 │
└──────────────────────────────────────┘
    ↓ metrics
┌──────────────────────────────────────┐
│ RESULTS EXPORT                       │
├──────────────────────────────────────┤
│ Display in terminal (Rich format)    │
│ Save to JSON file                    │
│ Export statistics report              │
└──────────────────────────────────────┘
    ↓
FINAL RESULTS + STATISTICS
```

---

## 🎯 Design Decisions

### **1. Two-Level LangGraph Architecture**

**Inner Graph** (Single Q/A): Handles one question
- Simpler state management
- Easier to debug individual queries
- Reusable for both single queries and benchmarks

**Outer Graph** (Benchmark): Manages multiple questions
- Loops over inner graph invocations
- Aggregates statistics and grades
- Exports results

**Why?**
- ✅ Separation of concerns
- ✅ Reusability (inner works standalone)
- ✅ Easier to extend (can add pre/post-processing)
- ✅ Statistics at both levels

---

### **2. Subgraph Pre-Compilation**

```python
# Pre-compile once during init
agent_workflow = ConcreteAgentWorkflow(...)
super().__init__(console, agent_workflow)  # Passes pre-compiled subgraph

# Reuse in every node call (no recompilation)
def run_agentic_workflow(self, state):
    sub_input = {...}
    final_sub_state = self.agent.invoke(sub_input)  # ← self.agent is pre-compiled
```

**Why?**
- ✅ Avoid recompiling 50x for 50 benchmark questions
- ✅ Faster execution
- ✅ Consistent graph structure

---

### **3. Stateless Distillation**

Each distillation call resets LLM context:
```python
self.llm.reset_context()  # Fresh start
response = self.llm.generate_response(...)
```

**Why?**
- ✅ Zero hallucination carryover
- ✅ Deterministic results
- ✅ Error isolation

---

### **4. Evidence Accumulation Cache**

PriorEvidenceTool enables:
```
Tool A finds fact → Store with ID "ev_abc"
Tool B references fact → Retrieve from cache (no re-search)
```

**Why?**
- ✅ Avoid redundant retrievals
- ✅ Faster loop iterations
- ✅ Traceable evidence chain

---

## 📚 Related Documentation

- [RAG Pipeline](../rag/RAG.md) - Retrieval architecture
- [Agentic Workflow](agent_workflow/AGENTIC_WORKFLOW.md) - Tool selection & planning
- [Benchmarks](benchmarks/BENCHMARKS.md) - Evaluation metrics
- [Answer Validation](answer_validation.py) - Path verification
- [Statistics](stats_collector.py) - Metrics collection

---

**Production-grade LangGraph orchestration for complex code Q&A** 🚀