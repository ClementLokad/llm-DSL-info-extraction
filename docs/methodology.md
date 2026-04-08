# Methodology

## 1. Data Preparation

The pipeline starts from Envision source scripts and applies parser-driven preprocessing to preserve DSL structure.
Chunking is not purely token-based; it is designed to keep semantically coherent code units.

## 2. Hybrid Retrieval Strategy

The system combines two complementary retrieval modes:

- **Semantic retrieval (RAG):** for conceptual questions and broader logic exploration.
- **Lexical retrieval (GREP):** for exact paths, identifiers, variables, or syntax-level constraints.

A routing stage determines which mode should be prioritized for each query.

## 3. Agentic Orchestration

The core workflow is implemented as a graph-based agentic pipeline:

- planning and tool selection;
- iterative evidence gathering;
- context aggregation;
- generation and post-cleaning of final answer.

The workflow supports controlled retries and conditional loops to improve answer quality.

## 4. Benchmarking and Quality Control

The project includes benchmark modes and grading workflows, with support for:

- similarity-based scoring;
- judge-model based evaluation;
- operational tracking such as latency and retrieval behavior.

## 5. Configuration and Reproducibility

The execution behavior is controlled through centralized configuration (`config.yaml`), including:

- model and rate-limit settings;
- chunking and retrieval parameters;
- index strategy and benchmark modes.
