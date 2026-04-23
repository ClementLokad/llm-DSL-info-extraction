# PSC INF01 X24 : LLM for Information Extraction in a DSL : Case of Envision (LOKAD)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Retrieval-Qdrant%20%2B%20Grep%20%2B%20Graph-0ea5e9.svg)](https://qdrant.tech/)
[![PSC Public Page](https://img.shields.io/badge/PSC-Public%20Page-22c55e.svg)](https://kpihx.github.io/envision-copilot-presentation/)

> Hybrid and agentic knowledge-extraction system for Lokad's proprietary Envision DSL.

## Project links

- Main repository: [github.com/ClementLokad/llm-DSL-info-extraction](https://github.com/ClementLokad/llm-DSL-info-extraction)
- PSC public page: [kpihx.github.io/envision-copilot-presentation](https://kpihx.github.io/envision-copilot-presentation/)

## Project overview

This repository contains the final X24 PSC prototype built to answer technical questions on a real Envision codebase used at Lokad.

The core difficulty is that Envision is a proprietary DSL: a large model cannot be trusted to understand the language, the file conventions, or the project structure from pretraining alone. The system therefore relies on **evidence-driven retrieval** instead of direct free-form code interpretation.

The final architecture combines:

- **semantic retrieval** for conceptual and business questions;
- **lexical retrieval** for exact paths, symbols, and syntax motifs;
- **structural graph navigation** for dependencies between scripts, folders, functions, tables, and data files;
- **an agentic loop** that decides which tool to use next based on the evidence already collected.

## Main capabilities

- **Parse** Envision `.nvn` scripts into 12 block types (imports, reads, writes, constants, tables, assignments, etc.)
- **Index** semantically: full chunks (1084 chunks from 6,078 blocks), LLM summaries, or RAPTOR hierarchies
- **Search** via hybrid retrieval: dense embeddings (semantic) + sparse BM25 (syntax) with RRF fusion
- **Route** with agentic planner: 7 tools (RAG, Grep, Graph, Script Finder, Tree, Prior Evidence, Distillation)
- **Validate** answers: lightweight path verification against codebase mapping
- **Benchmark** with 5 strategies: cosine similarity, dual cross-encoder, LLM judges, or hybrid
- **Collect** comprehensive statistics: tool call counts, LLM timing, retrieval latency, token usage

## System Architecture

### Entry Points

- **Interactive**: `python main.py` — single queries with LangGraph orchestration
- **Benchmark**: `python main.py --benchmark questions.json` — evaluate test suites with aggregated metrics
- **Configuration**: All behavior externalized to `config.yaml` (models, tools, benchmarks, validation)

### Core Layers

```
User Query
    ↓
Pipeline (LangGraph orchestration)
    ├─ Single Q/A or Benchmark loop
    ├─ Agentic Workflow (Strategic Planner + 7 Tools)
    ├─ Answer Validation (Path verification)
    └─ Grading (5 benchmark strategies)
    ↓
Results + Statistics (JSON export, terminal display)
```

### Documentation

- **[RAG Pipeline](rag/RAG.md)** — Parser → Chunker → Embedder (hybrid dense+sparse) → Retriever (RRF fusion)
- **[Agentic Workflow](pipeline/agent_workflow/AGENTIC_WORKFLOW.md)** — Mistral tool-calling planner with 7 tools and distillation
- **[Pipeline Orchestration](pipeline/PIPELINE.md)** — LangGraph states, nodes, and two-level architecture (single Q/A + benchmark)
- **[Benchmarks](pipeline/benchmarks/BENCHMARKS.md)** — 5 evaluation strategies with strengths/weaknesses comparison
- **[Agents](agents/AGENTS.md)** — LLM provider integrations (Claude, Mistral, Deepseek, Groq, Qwen)

### Tools Available

- **`rag_tool`** — Semantic search via hybrid embeddings (dense + sparse) with query transformation (HyDE/Fusion) and reranking
- **`grep_tool`** — Regex pattern matching on parsed blocks with block-type filtering
- **`graph_tool`** — Structural navigation for dependencies, imports, and relationships
- **`script_finder_tool`** — Full file reading (high token cost, use sparingly)
- **`tree_tool`** — Codebase structure with smart token-aware condensation
- **`prior_evidence_tool`** — Reuse prior findings via evidence cache (avoid re-searching)
- **`distillation_tool`** — Batch LLM-based fact extraction with stateless context (zero hallucination)


## Answer Validation

The pipeline includes a **lightweight non-blocking validation layer** for cited file paths:

**Configuration** (in `config.yaml`):
```yaml
main_pipeline:
  answer_validation:
    ignore_extension: true              # /path/script.nvn ≈ /path/script
    allow_partial_suffix_match: true    # folder/script ≈ /full/folder/script
    ignore_leading_slash: true          # /path ≈ path
    ignore_data_extensions: true        # Exclude .ion, .csv
    ignored_path_extensions: ["ion", "csv"]
```

**Validation Logic**:
- Extracts paths from final answers using 4 regex patterns
- Normalizes and matches against `mapping.txt`
- Tolerates formatting variations (spaces, slashes, extensions)
- If validation fails, triggers optional regeneration before returning warning

**See**: [pipeline/answer_validation.py](pipeline/answer_validation.py)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **API key errors** | Check `.env` file and `config.yaml` for correct model names |
| **Index not found** | Run `python build_index.py` before querying |
| **Out of memory** | Use `--indextype summary` or `--indextype raptor` for faster retrieval |
| **Slow queries** | Enable `--verbose` to see tool execution times; consider smaller `top_k` in config |
| **Benchmark failures** | Check benchmark JSON format matches expected schema (question, llm_response, reference) |
| **Path validation warnings** | These are non-blocking; review citations in final answer text |
| **LLM timeouts** | Increase `rate_limit_delay` in config; use faster models (Mistral vs Claude) |

**For detailed debugging**: Use `--verbose` flag to see:
- Planner decision reasoning
- Tool execution results
- LLM generation times
- Statistics (tool call counts, latencies)

## Repository structure

```text
llm-DSL-info-extraction/
├── main.py                      # Entry point: single queries & benchmarks
├── config.yaml                  # Centralized configuration
├── mapping.txt                  # File ID → original path mapping
├── requirements.txt             # Python dependencies
│
├── build_index.py              # Build full semantic index (1084 chunks)
├── build_summary_index.py       # Build LLM-summarized index (summaries)
├── build_raptor_index.py        # Build RAPTOR hierarchical index
│
├── agents/                      # LLM provider integrations (Claude, Mistral, etc.)
├── rag/                         # Retrieval-Augmented Generation pipeline
│   ├── parsers/                 # Envision script parsing (12 block types)
│   ├── chunkers/                # Semantic chunking with overlap
│   ├── embedders/               # Hybrid embedding (dense + sparse)
│   ├── retrievers/              # Vector search with RRF fusion
│   ├── summarizers/             # LLM-based chunk summarization
│   ├── query_transformers/      # Query enhancement (HyDE, Fusion)
│   ├── core/                    # Base classes and session management
│   └── utils/                   # Token handling, DB switching, script scanning
│
├── pipeline/                    # LangGraph orchestration
│   ├── agent_workflow/          # Agentic planner with 7 tools
│   ├── benchmarks/              # 5 evaluation strategies
│   ├── PIPELINE.md              # LangGraph orchestration docs
│   ├── langgraph_base.py        # State schemas and base classes
│   ├── answer_validation.py     # Path verification
│   ├── stats_collector.py       # Metrics collection
│   └── stats_reporter.py        # Results formatting
│
├── env_graph/                   # Dependency graph API (structural queries)
├── env_scripts/                 # Envision codebase (6,078 blocks → 1,084 chunks)
├── utils/                       # Config management & file mapping
├── docs/                        # Documentation & architecture diagrams
└── data/
    ├── faiss_index/             # Full semantic index (all chunks)
    ├── faiss_summary_index/     # Summary-based index (LLM summaries)
    ├── raptor_summary_index/    # Hierarchical RAPTOR index
    ├── qdrant/                  # Vector DB (alternative to FAISS)
    ├── grep_index/              # Parsed blocks cache (for grep_tool)
    ├── benchmark_results/       # Saved benchmark runs (JSON + stats)
    └── fastembed_models/        # Cached embeddings & models
```

## Quick start

### 1. Install dependencies

```bash
git clone git@github.com:ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the environment

- Copy `.env.example` to `.env` and set API keys (Claude, Mistral, etc.)
- Review `config.yaml` for critical settings:
  - **`agent`**: LLM provider and rate limiting
  - **`embedder`**: Dense model (all-MiniLM-L6-v2) and sparse model (Qdrant/bm25)
  - **`retrieval`**: Top-K chunks, rerank multiplier, RRF parameter
  - **`main_pipeline`**: Tools, planner, benchmarks, validation
  - **`paths`**: Input scripts, output data locations

### 3. Build Indexes

Three indexing strategies are available (choose one or build all):

#### **Full Semantic Index** (Recommended for most use cases)
```bash
python build_index.py
```
- Embeds all 1,084 semantic chunks into Qdrant + FAISS
- Hybrid search: dense (semantic) + sparse (BM25) with RRF fusion
- Output: `data/faiss_index/` and `data/qdrant/`
- Use case: Balanced cost/quality for code Q&A

#### **LLM-Summarized Index** (For denser representation)
```bash
python build_summary_index.py
```
- Generates ~100-word LLM summaries for each chunk
- Embeds summaries instead of raw chunks
- Output: `data/faiss_summary_index/` and `data/qdrant/codebase_rag_summary`
- Use case: Better semantic clustering, reduced noise
- Trade-off: Slight information loss, better for conceptual questions

#### **RAPTOR Hierarchical Index** (For complex codebases)
```bash
python build_raptor_index.py
```
- Builds hierarchical tree of summaries (bottom-up clustering)
- Combines retrieval at multiple abstraction levels
- Output: `data/raptor_summary_index/`
- Use case: Very large codebases, need context at multiple scales
- Trade-off: Higher latency, better for deep code understanding

**Pro Tip**: Start with `build_index.py`, then add summaries later if needed.

### 4. Ask Questions

#### **Single Query**
```bash
python main.py --query "How is revenue calculated in Items table?"
```

#### **Agentic Mode** (With tool routing, retries, distillation)
```bash
python main.py --agentic --query "Where is StockEvol defined and reused?"
```

#### **Verbose Output** (Debug mode)
```bash
python main.py --agentic --verbose --query "..."
```

#### **Benchmark Mode** (Evaluate on test suite)
```bash
python main.py --benchmark questions.json --benchmarktype hybrid --agentic
```

### Useful Flags

| Flag | Purpose |
|------|---------|
| `--agentic` | Enable full LangGraph workflow with tool routing |
| `--verbose` | Show detailed execution trace (LLM calls, tool results, statistics) |
| `--quiet` | Minimal output (final answer only) |
| `--query <str>` | Ask a single question and exit |
| `--agent <model>` | Override LLM (claude, mistral, groq, qwen) |
| `--indextype <type>` | Choose index (full_chunk, summary, raptor) |
| `--benchmark <file>` | Path to benchmark JSON file |
| `--benchmarktype <type>` | Evaluation metric (cosine, dual, llm, llm2, hybrid) |
| `--benchmarkagent <model>` | LLM for grading (separate from main agent) |

## Benchmarking

The project supports multiple benchmark modes:

```bash
# Hybrid (deterministic + neural) - RECOMMENDED
python main.py --benchmark questions.json --benchmarktype hybrid --agentic

# LLM Judge (binary 0/1)
python main.py --benchmark questions.json --benchmarktype llm --agentic

# LLM Judge (1-5 scale)
python main.py --benchmark questions.json --benchmarktype llm2 --agentic

# Dual Cross-Encoder (NLI + relevance)
python main.py --benchmark questions.json --benchmarktype dual

# Cosine Similarity (fast baseline)
python main.py --benchmark questions.json --benchmarktype cosine
```

Results are saved to:
- `data/benchmark_results/questions_TIMESTAMP.json` — Detailed grades + reasoning
- `data/benchmark_results/stats_TIMESTAMP.json` — Tool counts, LLM timing, latencies

See [benchmarks/BENCHMARKS.md](pipeline/benchmarks/BENCHMARKS.md) for comparison and guidance on metric selection.

## Key Metrics & Statistics

**Production Codebase**:
- **6,078** Envision code blocks parsed (12 block types)
- **1,084** semantic chunks after overlap-aware grouping (~5.6 blocks per chunk)
- **384-dimension** dense embeddings (all-MiniLM-L6-v2)
- **Sparse BM25** for keyword-level search
- **Reciprocal Rank Fusion** combining dense + sparse rankings

**Agentic System**:
- **7 tools** available: RAG, Grep, Graph, Script Finder, Tree, Prior Evidence, Distillation
- **3 LLMs** in pipeline: Planner (Mistral), Solver (Claude), Cleaner (Claude), Distiller (configurable)
- **Max 2 retries** per query (configurable via `max_retries`)
- **Batch distillation**: Extract facts from 3 results in 1 LLM call (not 3)

**Performance (Approximate)**:
- RAG hybrid retrieval: ~100-200ms per query
- Agentic workflow (full loop): ~20-40s (includes 3-5 LLM calls)
- Benchmark (50 questions): ~15-20 minutes with OpenAI models
- Path validation: ~5-10ms per answer

## Component Documentation

| Component | Purpose | Documentation |
|-----------|---------|---|
| **RAG Pipeline** | Parsing → Chunking → Embedding → Retrieval | [rag/RAG.md](rag/RAG.md) |
| **Parser** | Block type identification & dependency extraction | [rag/parsers/PARSER.md](rag/parsers/PARSER.md) |
| **Chunker** | Semantic chunking with overlap | [rag/chunkers/CHUNKERS.md](rag/chunkers/CHUNKERS.md) |
| **Embedder** | Hybrid dense + sparse embedding | [rag/embedders/EMBEDDERS.md](rag/embedders/EMBEDDERS.md) |
| **Retriever** | Vector search with RRF fusion | [rag/retrievers/RETRIEVERS.md](rag/retrievers/RETRIEVERS.md) |
| **Query Transformer** | HyDE & Fusion query enhancement | [rag/query_transformers/QUERY_TRANSFORMERS.md](rag/query_transformers/QUERY_TRANSFORMERS.md) |
| **Agentic Workflow** | Strategic planner + 7 tools | [pipeline/agent_workflow/AGENTIC_WORKFLOW.md](pipeline/agent_workflow/AGENTIC_WORKFLOW.md) |
| **Pipeline Orchestration** | LangGraph states & two-level architecture | [pipeline/PIPELINE.md](pipeline/PIPELINE.md) |
| **Benchmarks** | 5 evaluation strategies comparison | [pipeline/benchmarks/BENCHMARKS.md](pipeline/benchmarks/BENCHMARKS.md) |
| **Utilities** | Config & file mapping | [utils/UTILS.md](utils/UTILS.md) |

## Documentation site

The `docs/` directory contains the PSC public page and supporting material. The public entry point is:

- [kpihx.github.io/envision-copilot-presentation](https://kpihx.github.io/envision-copilot-presentation/)

The local Overleaf mirror used for the final report lives under `docs/PSC Rapport Final/` and is intentionally ignored by Git in this repository.

## Contribution notes

- Keep `config.yaml` as the single source of truth for runtime behavior.
- Prefer updating parser-, retrieval-, or workflow-specific modules rather than introducing duplicated logic.
- If you touch the public-facing project story, keep both `README.md` and `docs/README.md` aligned.

## License

This project is distributed under the private license included in [LICENSE](LICENSE).
