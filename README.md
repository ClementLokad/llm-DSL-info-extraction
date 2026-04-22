# Envision PSC X24

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Retrieval-Qdrant%20%2B%20Grep%20%2B%20Graph-0ea5e9.svg)](https://qdrant.tech/)
[![PSC Public Page](https://img.shields.io/badge/PSC-Public%20Page-22c55e.svg)](https://kpihx.github.io/envision-copilot-presentation/)

> Hybrid and agentic knowledge-extraction system for Lokad's proprietary Envision DSL.

## Public deliverables

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

- Parse Envision `.nvn` scripts into structured blocks.
- Build semantic indexes over full chunks, summaries, or RAPTOR summaries.
- Search exact code patterns through parsed Envision blocks with `grep_tool`.
- Navigate script/data/function dependencies with `graph_tool`.
- Run an agentic workflow with retries, tool routing, and final answer cleaning.
- Benchmark the system with multiple judge modes.
- Apply a lightweight non-blocking validation pass on cited script paths in final answers.

## System architecture

Two entry modes coexist:

1. `python main.py`
   Use the configured defaults for interactive or one-shot questioning.
2. `python main.py --agentic`
   Enable the full LangGraph-based workflow with tool routing and iterative evidence gathering.

The agentic stack currently revolves around these tools:

- `rag_tool`
  Semantic retrieval on indexed Envision chunks.
- `grep_tool`
  Exact search on parsed Envision blocks, optionally filtered by source path and block type.
- `graph_tool`
  Structural navigation over the dependency graph built from scripts and data relationships.
- `tree_tool`
  Compact hierarchy view used to orient the planner before deeper searches.
- `script_finder_tool`
  Mapping-aware path recovery from numeric mirrored filenames.

## Lightweight answer validation

The agentic mode can include a first non-blocking validation layer configured in `main_pipeline.answer_validation` inside `config.yaml`.

Its current scope is intentionally narrow:

- it validates only **script paths** cited in final answers;
- it checks them against `mapping.txt`;
- it tolerates missing leading slashes, extra spaces, missing extensions, and `.nvn` / `.nvm` confusion;
- it can accept a unique partial suffix match;
- if suspicious paths are detected, it triggers a short regeneration attempt before returning a warning section.

Data files such as `.ion` or `.csv` are deliberately ignored in this V1 because `mapping.txt` only covers scripts. Proper validation of data paths should later rely on `env_graph`.

## Repository structure

```text
envision/
├── main.py
├── config.yaml
├── build_index.py
├── build_summary_index.py
├── build_raptor_index.py
├── mapping.txt
├── env_graph/
├── pipeline/
│   ├── agent_workflow/
│   └── benchmarks/
├── rag/
│   ├── core/
│   ├── parsers/
│   ├── chunkers/
│   ├── embedders/
│   ├── summarizers/
│   └── retrievers/
├── env_scripts/
├── docs/
└── data/
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

- Copy `.env.example` to `.env` and set the API keys you need.
- Review `config.yaml` before running the pipeline.

Useful sections:

- `agent`
- `embedder`
- `retriever`
- `main_pipeline`
- `benchmark`

### 3. Build indexes

```bash
python build_index.py
python build_summary_index.py
python build_raptor_index.py
```

### 4. Ask questions

```bash
python main.py --query "Which scripts read /Clean/Items.ion?"
python main.py --agentic --query "Where is StockEvol defined and reused?"
python main.py --agentic --verbose
```

Useful flags:

- `--agentic`
- `--verbose`
- `--quiet`
- `--query`
- `--agent`
- `--indextype`
- `--fusion`
- `--benchmarkpath`
- `--benchmarktype`
- `--benchmarkagent`

## Benchmarking

The project supports several benchmark modes driven by `config.yaml` and CLI flags.

Examples:

```bash
python main.py --benchmarkpath questions.json --benchmarktype llm_as_a_judge
python main.py --benchmarkpath questions.json --benchmarktype hybrid
```

## Documentation site

The `docs/` directory contains the PSC public page and supporting material. It is published at:

- [kpihx.github.io/envision-copilot-presentation](https://kpihx.github.io/envision-copilot-presentation/)

The local Overleaf mirror used for the final report lives under `docs/PSC Rapport Final/` and is intentionally ignored by Git in this repository.

## Contribution notes

- Keep `config.yaml` as the single source of truth for runtime behavior.
- Prefer updating parser-, retrieval-, or workflow-specific modules rather than introducing duplicated logic.
- If you touch the public-facing project story, keep both `README.md` and `docs/README.md` aligned.

## License

This project is distributed under the private license included in [LICENSE](LICENSE).
