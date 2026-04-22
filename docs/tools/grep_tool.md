# Tool: Grep Tool (Lexical Search)

The `grep_tool` enables exact text searches across all project files.

## 📝 Role
While RAG searches by meaning, the `grep_tool` searches by exact character strings. It is used by the agent to precisely locate variables, constants, or specific function calls when the name or pattern is known.

## 📂 Technical Details
- **Source Code**: [`grep_tool.py`](https://github.com/ClementLokad/llm-DSL-info-extraction/tree/main/pipeline/agent_workflow/grep_tool.py)
- **Operation**: Recursive filesystem traversal filtering for `.nvn` extensions and block types (table, variable, etc.).

## 💡 Concrete Examples
Based on the extraction needs identified in `questions.json`:
- **Action**: Locate all occurrences of the `ReDispatchCycle` field to identify its configuration.
- **Action**: Find the script performing consistency checks on the `FuturSellPrice` column.
- **Question**: *"Which scripts use the RedispatchCycleWeeks variable?"* (The agent uses grep to find the exact string).

---
> [!NOTE]
> Combined with the **`graph_tool`**, it allows transitioning from a text search to full structural navigation.
