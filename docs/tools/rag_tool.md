# Tool: RAG Tool (Semantic Search)

The `rag_tool` is the interface between the agent and the project's vector knowledge base.

## 📝 Role
This tool allows the agent to perform semantic searches across technical documentation and Envision source code. It is essential for understanding business concepts and complex DSL syntax without requiring exact matches.

## 🏛️ Integration
The `rag_tool` relies directly on the data pipeline located in the **[`rag/`](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/envision/rag/)** directory.
This pipeline handles:
- **Parsing** of source files.
- **Hybrid Chunking** (strategy optimized for code).
- **Indexing** in a vector database (Qdrant/FAISS).

## 📂 Technical Details
- **Source Code**: [`rag_tool.py`](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/envision/pipeline/agent_workflow/rag_tool.py)
- **Logic**: Receives a natural language query, transforms it into an embedding, and retrieves the most relevant chunks.

## 💡 Concrete Examples
Extracted from the `questions.json` test file:
- **Question**: *"Is there a function to calculate stock evolution over time considering arrivals and demand?"*
- **Question**: *"Where are the best sellers calculated?"*
- **Question**: *"Is there a place to find condensed information for analyzing different suppliers?"*

---
> [!TIP]
> RAG is particularly useful at the beginning of a session to guide subsequent, more precise lexical searches (Grep).
