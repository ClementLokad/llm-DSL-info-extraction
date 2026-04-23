# 🤖 Agentic Workflow - Advanced RAG Agent with Mistral Tool-Calling

> Production agentic orchestration system for complex Envision codebase question-answering. Strategic planning with multi-tool retrieval, intelligent reranking, and LLM-based distillation.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUESTION                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   AgenticPipeline (LangGraph)    │
        │  ┌────────────────────────────┐  │
        │  │ Agentic Workflow (Subgraph)│  │ ◄─── Main agent loop
        │  │                            │  │
        │  │ ┌──────────────────────┐   │  │
        │  │ │  Planner (Mistral)   │   │  │  Strategic decision
        │  │ │ + Tool-Calling API   │   │  │
        │  │ └──────────┬───────────┘   │  │
        │  │            │               │  │
        │  │   ┌────────┴───────┐       │  │
        │  │   ▼               ▼        │  │
        │  │ ┌────────┐ ┌────────┐      │  │
        │  │ │  Tool  │ │  Tool  │ ...  │  │
        │  │ └────────┘ └────────┘      │  │
        │  │   (RAG, Grep, Graph,       │  │
        │  │    Script, Tree, Prior,    │  │
        │  │    Distillation)           │  │
        │  │                            │  │
        │  │ ┌──────────────────────┐   │  │
        │  │ │ Distillation (Batch) │   │  │
        │  │ │  Extract facts       │   │  │
        │  │ └──────────────────────┘   │  │
        │  └────────────────────────────┘  │
        │                                  │
        │  ┌────────────────────────────┐  │
        │  │   Solver (Main LLM)        │  │ Generate answer from facts
        │  │   + System Prompt          │  │
        │  └────────────────────────────┘  │
        │                                  │
        │  ┌────────────────────────────┐  │
        │  │   Cleaner (Cleaning LLM)   │  │ Format final answer
        │  │   + System Prompt          │  │
        │  └────────────────────────────┘  │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │    Answer Validation             │
        │  Path verification & formatting  │
        └──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FINAL ANSWER                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Core Components

### 1. **AgenticPipeline** (`agentic_pipeline.py`)

Main LangGraph pipeline orchestrating the entire workflow.

**Responsibilities**:
- Manages sub-graph invocation (ConcreteAgentWorkflow)
- Invokes Solver LLM for answer generation
- Invokes Cleaner LLM for formatting
- Handles loop control (retry logic)
- Validates answer sources

**Key Nodes**:
```
run_agentic_workflow
    ↓
check_agent_logic (Decide: more search or generate?)
    ↓
decide_after_logic_check (Route: regenerate or clean)
    ↓ [if regenerate]
run_agentic_workflow (loop)
    ↓ [if ready]
clean_generated_answer
    ↓
validate_paths
    ↓
FINAL_ANSWER
```

**Configuration**:
```yaml
main_pipeline:
  agent_logic:
    main_llm: "deepseek-r1"      # Solver LLM
    cleaning_llm: "deepseek-v3"  # Cleaner LLM
    planner_llm: "mistral"       # Strategic planner
    max_retries: 2               # Max loop iterations
  answer_validation:
    ignore_extension: true
    allow_partial_suffix_match: true
```

---

### 2. **ConcreteAgentWorkflow** (`concrete_workflow.py`)

Implements the **Strategic Planner** using Mistral native tool-calling.

**Key Design Choice**: **Mistral Tool-Calling API** instead of XML parsing
- ✅ Native JSON tool_calls from model (no parsing errors)
- ✅ First-class JSON schema validation
- ✅ Better instruction-following with structured output
- ✅ Coherent tool conversation history

**Workflow Loop**:
```
1. Kickoff Prompts (First tool selection)
   - Question context
   - Codebase structure (FileTreeTool)
   - Verified facts (knowledge bank)
   ↓
2. Planner.generate_with_tools()
   - Mistral selects next tool
   - Returns JSON tool_call
   ↓
3. Tool Execution
   - Execute selected tool
   - Collect results
   ↓
4. Distillation (Batch)
   - Extract facts from raw results
   - Track evidence IDs
   ↓
5. Continuation Prompts (Keep searching or done?)
   - Question + Proposed answer
   - Previous history
   - Decision logic
   ↓
6. Loop Decision
   - If insufficient: call next tool
   - If sufficient: submit_answer
```

**State Machine**:
```python
WorkflowState = {
    "pipeline_state": {...},           # Question, knowledge_bank, history
    "regenerate": bool,                # Need more info?
    "error": str | None,               # Error message
    "rewritten_prompt": str,           # Agent's own rewording of question
    "current_thought": str,            # Planner's reasoning
    "pending_tool_call": {...},        # Next tool to execute
    "accumulated_evidence": {...},     # Evidence cache (ID → RetrievalResult)
}
```

---

## 🛠️ Tools Available (7 Total)

### **Tool 1: RAG Tool** (SimpleRAGTool)

**Purpose**: Semantic search via hybrid embeddings.

**Implementation Choice**: **Hybrid Dense + Sparse with RRF**
- Dense embedder (all-MiniLM-L6-v2): Semantic understanding
- Sparse embedder (Qdrant/bm25): Keyword matching
- **RRF Fusion**: Combine rankings with formula: `score = 1/(k+rank_dense) + 1/(k+rank_sparse)`

**Advanced Features**:
- Query transformation (optional): HyDE or Fusion
- Reranking with keywords: Boost chunks containing searched keywords
- Source filtering: Penalize results not matching source_regex
- Constant replacement: Replace Envision constants before returning
- Cross-encoding reranker: Optional fastembed reranker for precision

**Usage**:
```python
tool.retrieve(
    query="Calculate total revenue?",
    top_k=5,
    key_words=["Items", "Revenue"],        # For reranking boost
    sources="items.*\\.nvn$",               # Source filter (regex)
    rerank_multiplier=2,                   # Pre-fetch 2x, rerank to top-k
    verbose=True
)
# Returns: List[RetrievalResult] (sorted by RRF score)
```

**Why Multiple Features?**
- Hybrid search: Captures both semantic + syntactic relevance
- Reranking: Prioritizes chunks with exact keywords (higher precision)
- Source filtering: Narrows to relevant script families
- Query transformation: Generates multiple angles of approach
- Constant replacement: Makes results readable (Envision-specific)

---

### **Tool 2: Grep Tool** (GrepTool)

**Purpose**: Precise regex-based pattern matching on loaded index.

**Key Advantage**: No embedding needed, exact pattern matching.

**Advanced Features**:
- Regex pattern search: Full regex support
- Source filtering: Pre-compiled regex for speed
- Block type filtering: Filter by IMPORT, READ, WRITE, CONST, etc.
- Result truncation: Limit output to fit context budget
- Index persistence: Load/save pickle'd blocks for speed

**Usage**:
```python
tool.search(
    pattern=r"Items\.Revenue\s*=",         # Regex pattern
    source_regex=r".*items.*\.nvn$",       # Filter to source files
    bloc_type=[BlockType.ASSIGNMENT],      # Only ASSIGNMENT blocks
    warnings=[]                             # Track invalid regex
)
# Returns: List[RetrievalResult]
```

**When to Use?**
- Looking for exact syntax: `Items.Revenue =`
- Need specific block types: assignments, constants, imports
- Want to avoid semantic approximations
- Results must be precise (not "similar")

---

### **Tool 3: Graph Tool** (EnvisionGraphTool)

**Purpose**: Query structural graph for relationships & dependencies.

**Graph Queries**:
- `tree`: File/script hierarchy with max depth/children limits
- `node`: Get single node details (node_id)
- `neighbors`: Find nodes linked to a target (imports, calls, depends_on)
- `edges`: All edges of a type (import, dependency, calls, etc.)
- `search`: Full-text search within graph nodes

**Architecture**:
- Built from env_graph module (external dependency graph)
- Auto-builds graph if missing
- Safe serialization (JSON-compatible)
- Response truncation (12K chars default)

**Usage**:
```python
tool.execute(
    action="neighbors",
    node_id="script:items.nvn",
    direction="outgoing",          # incoming | outgoing | all
    relation_type="imports"         # Optional filter
)
# Returns: Dict with neighbor node details and relation info
```

**When to Use?**
- "Who imports this module?"
- "What are the dependencies of X?"
- "What's the structure of /folder/?"
- Need to understand code organization/topology

---

### **Tool 4: Script Finder Tool** (PathScriptFinder)

**Purpose**: Find and read entire scripts by name.

**Design Decision**: **Rare, high-token-cost operation**
- Only use when no alternative exists
- Prefer grep_tool with source filter instead
- Full file read may explode context

**Features**:
- Fuzzy script matching: Supports partial names, regex
- File mapping: Resolves script names to full paths
- Direct file read: Returns complete source
- Path tracking: Remembers original paths

**Usage**:
```python
tool.find_scripts(["items.nvn", "orders"])
# Returns: ["/absolute/path/to/items.nvn", ...]

content = tool.read_file("/absolute/path/to/items.nvn")
```

**Best Practices**:
- Use sparingly (token budget aware)
- Always try grep_tool with source filter first
- Reserve for "must read entire file" scenarios

---

### **Tool 5: Tree Tool** (FileTreeTool)

**Purpose**: Display codebase structure within token budget.

**Smart Condensation**:
- `max_depth`: Stop rendering past this level
- `max_children`: Show first N items, summarize rest
- `fit_tree_to_context()`: Auto-adjust parameters to stay within token limit

**Usage**:
```python
tree_str = tool.custom_tree(
    root_path="/",
    max_depth=3,
    max_children=10
)
# Output:
# /
# ├── env_scripts/
# │   ├── items.nvn
# │   ├── orders.nvn
# │   └── ... (50 more items)
# ├── pipeline/
# │   ├── agent_workflow/
# │   │   ├── agentic_pipeline.py
# │   │   └── ...
```

**Why Important?**
- Bootstrap context: Initial structure awareness
- Query grounding: Planner understands what exists
- Source validation: Confirms scripts exist before searching

---

### **Tool 6: Prior Evidence Tool** (PriorEvidenceTool)

**Purpose**: Retrieve and reuse facts from previous tool calls.

**Design Decision**: **Evidence Accumulation Cache**
- Each tool result gets a stable ID (hash-based)
- IDs tracked in `accumulated_evidence` dict
- Planner can reference prior findings without re-searching

**Features**:
- `retrieve_prior_evidence(ids)`: Bulk retrieve by ID list
- `format_results_by_source()`: Group results by source file
- Explicit tracking: Reports missing IDs
- Zero re-work: Reuse previous results across tool loops

**Usage**:
```python
# Store in state
accumulated_evidence = {
    "ev_a1b2c3": RetrievalResult(...),
    "ev_x9y8z7": RetrievalResult(...),
}

# Retrieve
prior = tool.retrieve_prior_evidence(["ev_a1b2c3", "ev_x9y8z7"])
# Returns: {
#   "ev_a1b2c3": [RetrievalResult(...)],
#   "ev_x9y8z7": [RetrievalResult(...)],
# }
```

**When to Use?**
- Question requires multiple related concepts
- Need to cross-reference previous findings
- Avoid duplicate retrieval (token efficiency)

---

### **Tool 7: Distillation Tool** (LLMDistillationTool)

**Purpose**: Extract and preserve key facts from raw documents.

**Design Choice**: **Batch Distillation + Stateless LLM**
- Distill multiple items in one LLM call (save tokens)
- Reset LLM context before each distillation (avoid hallucination)
- Use system role for static persona (matches training)
- Use user role for dynamic content (query, documents, generation)

**Two Modes**:

**Single Distillation** (for small content):
```python
fact = tool.distill(
    content="Items.Revenue = Items.Quantity * Items.Price",
    query="How to calculate revenue?",
    thought="Looking for revenue calculation formulas",
    source="items.nvn"
)
# Returns: "Revenue calculated by multiplying quantity and price"
```

**Batch Distillation** (for efficiency):
```python
facts = tool.distill_batch(
    items=[
        ("Content A", "source_a.nvn"),
        ("Content B", "source_b.nvn"),
        ("Content C", "source_c.nvn"),
    ],
    query="How to calculate revenue?",
    thought="...",
    previous_generation="..."  # LLM's current answer
)
# Returns: List[(summary, evidence_id_list)]
```

**System Prompts** (Role-based):
- Memory Manager persona: Static, in system role
- Dynamic content: Query, documents, generation in user role
- Prevents mixing identity (persona) with data (content)

**Why Batch Mode?**
- ✅ Fewer LLM calls (3 items → 1 call instead of 3)
- ✅ Context awareness: LLM sees all items together
- ✅ Efficient: Batch processing is standard practice
- ✅ Cost reduction: Fewer API calls

---

## 🧠 Planner Strategy

### **Kickoff Phase** (First Tool Selection)

**Inputs**:
- User question
- Codebase structure (FileTreeTool output)
- Verified facts (knowledge_bank)

**Planner Decision Logic**:
```
IF question mentions specific file path, function name, or variable
  → Use grep_tool (precise, exact match)
   
ELSE IF question asks about concept, behavior, or business logic
  → Use rag_tool (semantic search)
   
ELSE IF question asks "who imports X?" or structure questions
  → Use graph_tool (dependency queries)
   
ELSE
  → Use rag_tool as default (catches most semantic queries)
```

**Kickoff Prompts Include**:
- Full question context
- Codebase structure tree (3-level max to save context)
- Bootstrap rule: "Prefer RAG for safety if structure is uncertain"
- Planning instructions: Tool selection criteria

---

### **Continuation Phase** (Keep Searching or Done?)

**Decision Logic**:
```
1. Read Proposed Solution from Solver LLM
   
2. CRITICAL CHECK:
   Does solution directly answer the question?
   - Must include concrete evidence (file paths, line numbers)
   - Must not be vague ("I don't know", "possibly", "might be")
   
3. DECISION:
   IF answer is concrete + complete
     → call submit_answer (stop loop)
   
   ELSE
     → Choose next tool based on:
        • What failed? (grep too many/few results → try RAG)
        • Alternative angles? (Try different keyword variants)
        • Need structure? (Switch to graph_tool)
        • Need full context? (Use script_finder_tool)
```

**Anti-Repetition**:
- Track previous tool calls
- Never use same tool + parameters twice
- Switch tool family if one fails (grep → RAG, or vice versa)

---

## 🔄 Request Flow - Step by Step

### **Complete Query Lifecycle**

```
1. USER QUESTION
   "How is revenue calculated in Items table?"
   
2. AGENTIC WORKFLOW (Subgraph Starts)
   └─ Planner LLM receives:
      • Question + context
      • Codebase structure
      • Kickoff prompts
   
3. PLANNER DECISION #1
   Selected tool: rag_tool
   Reason: "Semantic concept (revenue calculation) requires search
            beyond exact pattern matching"
   
4. RAG TOOL EXECUTION
   Query: "How is revenue calculated in Items table?"
   └─ Embed query (dense + sparse)
   └─ Search Qdrant (hybrid)
   └─ RRF fusion (combine results)
   └─ Rerank with keywords: ["revenue", "calculated"]
   └─ Results: [
        RetrievalResult(chunk="Items.Revenue = ...", score=0.92),
        RetrievalResult(chunk="quantity * price", score=0.87),
        RetrievalResult(chunk="Revenue table definition", score=0.81),
      ]
   
5. DISTILLATION TOOL (Batch)
   Items to distill: [result.chunk.content for each result]
   Query: "How is revenue calculated?"
   └─ LLM extracts facts:
      "Revenue = Quantity * Price per item"
      "Then sum across all items"
   
6. SOLVER LLM GENERATION
   System: Base instructions + Envision expertise
   User: Question + retrieved facts + distilled summaries
   └─ Generates answer:
      "Revenue in Items table is calculated by multiplying
       quantity by price for each item, then summing the results.
       See: items.nvn lines 45-50"
   
7. LOGIC CHECK
   Q: Does this answer the question concretely?
   A: Yes - specific formula + file reference
   Decision: → Proceed to cleaning
   
8. CLEANER LLM (Formatting)
   System: Formatting rules (structure, language, style)
   User: Raw answer
   └─ Cleans and reformats:
      "<final_answer>
        Revenue is calculated as Quantity × Price per item,
        then summed. (items.nvn:45-50)
       </final_answer>"
   
9. PATH VALIDATION
   └─ Verify "items.nvn:45-50" exists in codebase
   └─ Confirm path format
   
10. FINAL ANSWER
    Returned to user
```

---

## 🎯 Design Choices & Rationale

### **1. Mistral Tool-Calling API (vs XML Parsing)**

**Alternative**: Parse XML from LLM output
```xml
<tool name="rag_tool">
  <query>...</query>
  <sources>...</sources>
</tool>
```

**Why Mistral Native?**
- ✅ Zero parsing errors (structured JSON guarantee)
- ✅ Better instruction-following (native API training)
- ✅ Cleaner history (tool_calls role, not XML in text)
- ✅ Automatic validation (schema enforced by API)
- ❌ Less flexible (schema must be pre-defined)

---

### **2. Hybrid Retrieval with RRF**

**Alternative**: Use only dense embeddings

**Why Hybrid?**
- Dense: Catches semantic meaning ("revenue" → "income", "total")
- Sparse: Catches exact syntax (Items.Revenue → exact match)
- RRF: Fair ranking formula (doesn't favor one modality)
- Real-world: Code searches need both semantic + syntactic

**RRF Formula**:
```
score = 1/(k + rank_dense) + 1/(k + rank_sparse)
```
- k=60 (default): Balances early-ranking items
- Equal weight to both modalities
- Top dense + top sparse both valued

---

### **3. Batch Distillation**

**Alternative**: Distill items one-by-one

**Why Batch?**
- ✅ 1 LLM call for 3 items (saves 2 calls)
- ✅ LLM sees all items together (better context)
- ✅ Token efficient (batch overhead < 3 individual calls)
- ✅ Deterministic grouping (same results per batch)
- ❌ Slightly longer response (but acceptable)

**Cost Example** (per query):
```
One-by-one:    3 RAG results × 1 distillation call = 3 LLM calls
Batch mode:    3 RAG results → 1 batch distillation = 1 LLM call
               Savings: 66% fewer calls
```

---

### **4. Stateless Distillation LLM**

**Pattern**: `llm.reset_context()` before each distillation

**Why Reset?**
- ✅ Zero memory carryover (fresh start each time)
- ✅ Prevents hallucination drift (LLM doesn't "remember" prior docs)
- ✅ Consistent results (same input → same output)
- ✅ Error isolation (one bad fact doesn't poison future calls)
- ❌ Slightly more calls (reset overhead minimal)

---

### **5. Evidence Accumulation Cache**

**Use Case**: Multi-turn question investigation

**Without cache**:
```
Tool A finds fact: "X happens in items.nvn"
Tool B later needs same fact: Re-search, Re-retrieve (wasted tokens)
```

**With cache** (PriorEvidenceTool):
```
Tool A finds fact → Stored with ID "ev_abc123"
Tool B needs fact → Call prior_evidence_tool(["ev_abc123"])
                 → Returns cached result (zero cost)
```

**Benefits**:
- ✅ Reference previous findings
- ✅ Zero re-search cost
- ✅ Faster loop iterations
- ✅ Explicit ID tracking (debuggable)

---

## 📋 Configuration Reference

```yaml
# Planner & LLM configuration
main_pipeline:
  agent_logic:
    planner_llm: "mistral"         # Tool-calling engine
    main_llm: "claude"             # Answer generation
    cleaning_llm: "claude"         # Final formatting
    distillation_llm: "claude"     # Fact extraction
    max_retries: 2                 # Max search loops
    
# Tool configuration
main_pipeline:
  kickoff_tree_max_depth: 5        # Initial tree depth
  kickoff_tree_max_children: 20    # Items per folder
  grep_tool:
    index_path: "data/grep_index"
    case_sensitive: false
  graph_tool:
    config_path: "env_graph/config.yaml"
    auto_build: true
    default_domain: "scripts"
    response_max_chars: 12000
    
# RAG tool
rag:
  retrieval:
    type: "qdrant"
    collection_name: "codebase_rag"
    top_k_chunks: 5
    rerank_multiplier: 2
    k_for_rrf: 60
    
# Answer validation
main_pipeline:
  answer_validation:
    ignore_extension: true
    allow_partial_suffix_match: true
    ignore_leading_slash: true
    ignore_data_extensions: true
    ignored_path_extensions: ["ion", "csv"]
```

---

## 🚀 Entry Points

### **Building Index** (Offline)
```bash
python build_index.py
```

### **Query & Retrieval** (Online)
```python
from pipeline.agentic_pipeline import AgenticPipeline
from pipeline.agent_workflow.concrete_workflow import ConcreteAgentWorkflow

# Initialize
pipeline = AgenticPipeline(console, agent)
state = AgentGraphState(question="How to calculate revenue?", ...)

# Run
final_state = pipeline.run(state)
answer = final_state["final_answer"]
```

### **Benchmarking**
```bash
python -c "from pipeline.stats_reporter import generate_report; generate_report()"
```

---

## 🔍 Debugging Tips

### **1. Why Did Tool X Get Selected?**
- Check `state["current_thought"]` (planner's reasoning)
- Check `state["pending_tool_call"]` (tool name + arguments)
- Review continuation prompts for decision logic

### **2. Why Was Result Y Ranked Low?**
- Check RRF score: `1/(k+rank_dense) + 1/(k+rank_sparse)`
- Check reranking: keyword matches boost score
- Check source filtering: wrong source → penalized

### **3. Why Did Search Loop?**
- Check `state["regenerate_needed"]`
- Review `state["execution_history"]` (tool sequence)
- Check answer validation feedback

---

## 📚 Related Documentation

### Pipeline & Orchestration
- [LangGraph Pipeline](../PIPELINE.md) - Main LangGraph orchestration
- [LLM Agents](../../agents/AGENTS.md) - Agent implementations and configurations
- [Benchmarking Framework](../benchmarks/BENCHMARKS.md) - Performance evaluation

### RAG Components (Tools Used)
- [RAG Pipeline](../../rag/RAG.md) - Semantic retrieval architecture
- [Parser Documentation](../../rag/parsers/PARSER.md) - Code block extraction
- [Embedders Documentation](../../rag/embedders/EMBEDDERS.md) - Embedding models
- [Retrievers Documentation](../../rag/retrievers/RETRIEVERS.md) - Vector search
- [Query Transformers Documentation](../../rag/query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement
- [Chunkers Documentation](../../rag/chunkers/CHUNKERS.md) - Code chunking

### Utility Modules
- [Answer Validation](../answer_validation.py) - Path verification
- [Stats Collection](../stats_collector.py) - Metrics & benchmarking
- [Env Graph](../../env_graph/README.md) - Dependency graph building

### Getting Started
- [Quick Start Tutorial](../../TUTORIAL.md) - Setup and usage guide
- [Project README](../../README.md) - High-level architecture

---

**Production-grade agentic system for complex code Q&A** 🚀