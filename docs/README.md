# PSC INF01
> **LLM for Information Extraction in a DSL : Case of Envision (LOKAD)**


Welcome to the documentation for the Envision Copilot, an agentic system designed to navigate and explain Lokad's proprietary Envision DSL.

## 🏛️ Project Architecture
The system is built as a multi-tool agent capable of planning and executing complex search strategies over code and documentation.

- **[Full Architecture](architecture.md)**: Global overview of the ReAct pipeline and feedback loops.
- **[Semantic Pipeline (RAG)](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/rag/)**: The underlying engine for semantic indexing and retrieval. Detailed usage: **[RAG Tool](tools/rag_tool.md)**.
- **[Structural Engine (Graph)](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/env_graph/)**: Static analysis layer for building the dependency graph. Detailed usage: **[Graph Tool](tools/graph_tool.md)**.

## 🛠️ Primary Agentic Tools
The agent uses a combination of three specialized tools to solve user queries:

1. **[RAG Tool](tools/rag_tool.md)**: Semantic search on documentation and code chunks.
2. **[Graph Tool](tools/graph_tool.md)**: Structural navigation through dependencies (imports, reads, writes).
3. **[Grep Tool](tools/grep_tool.md)**: Lexical search for exact patterns and variables across `.nvn` files.

## 🚀 Key Features
- **Planning & Self-Correction**: The agent dynamically adapts its strategy based on the results of previous actions.
- **Evidence-Driven**: Every answer is grounded in specific code extracts or structural links.
- **DSL Specific**: Tailored parsers and analyzers for the Envision language.
