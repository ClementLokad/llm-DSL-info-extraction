# Tool: Graph Tool (Structural Navigation)

The `graph_tool` allows the agent to navigate the structural dependencies of the Envision codebase. It provides a multi-layer view of how scripts, data files, tables, and functions are interconnected.

## 📝 Role
This tool is used to answer questions about the structure and impact of changes (e.g., "Which scripts read this file?", "What are the dependencies of this module?"). It complements the RAG (semantic) and Grep (lexical) tools by providing a topological map of the project.

## 🛠️ Integration
This tool leverages the API developed in the **[`env_graph/`](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/envision/env_graph/)** directory.
The graph engine:
- Statically analyzes Envision scripts (`.nvn`).
- Builds a directed dependency graph (file I/O, imports, definitions).
- Provides a fast interface for the agent to skip irrelevant files.

## 📂 Technical Details
- **Source Code**: [`graph_tool.py`](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/envision/pipeline/agent_workflow/graph_tool.py)
- **Capacities**: 
  - `tree`: Explore folder/file hierarchy (scripts or data).
  - `node`: Get detailed metadata for a specific node (script, function, table).
  - `search`: Quickly locate a node by its name or path.
  - `neighbors`: Discover relationships (incoming/outgoing) filtered by type (`reads`, `writes`, `imports`, `defines`).
  - `edges`: Query global relationship patterns across the network.

## 💡 Concrete Examples
Taken directly from `questions.json`:
- **Impact Analysis**: *"Identify all scripts reading the file `/Clean/Orders.ion` to assess the impact of a schema change."*
  - **Action**: `neighbors(node_id="/Clean/Orders.ion", direction="incoming", relation_type="reads")`
- **Module Discovery**: *"Find all modules imported by `/1. utilities/DataLoader.nvn`."*
  - **Action**: `neighbors(node_id="/1. utilities/DataLoader.nvn", direction="outgoing", relation_type="imports")`
- **Definition Lookup**: *"What are the modules, scripts, and files on which the Forecasting script depends?"*
  - **Action**: `neighbors(node_id="/Forecasting/Forecasting.nvn", direction="outgoing")`

---
> [!IMPORTANT]
> The `graph_tool` significantly reduces token consumption by allowing the agent to target only relevant files based on their structural connections.
