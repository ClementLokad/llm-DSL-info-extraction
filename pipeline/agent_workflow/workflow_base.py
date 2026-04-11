import re
from abc import abstractmethod
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from langgraph.graph import END, StateGraph, START
from pipeline.langgraph_base import AgentGraphState, ActionLog, KnowledgeElement, format_knowledge_element
from rag.core.base_retriever import RetrievalResult, BaseRetriever
from rag.core.base_embedder import BaseEmbedder
from agents.prepare_agent import *
from config_manager import get_config

from rich.console import Console


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

def _tool_desc(name: str, description: str, properties: Dict[str, Any],
                  required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build a Mistral-compatible tool definition dict."""
    # Add thought property
    properties["thought"] = {
        "type": "string",
        "description": "Your concise reasoning for choosing this tool and these parameters."
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": (required or []) + ["thought"],
            },
        },
    }

class Tool:
    """Base Class for all tools"""
    def get_description(self) -> Dict[str, Any]:
        """Return a OpenAI-compatible tool schema for this tool."""
        return _tool_desc(name="base_tool", description='', properties={})


# ---------------------------------------------------------------------------
# Distillation tool
# ---------------------------------------------------------------------------

class BaseDistillationTool(Tool):
    """
    A base tool for distilling retrieved content into concise knowledge facts.
    Implements 'Distill and Discard' with batch processing to save tokens/calls.
    """
    def __init__(self, llm_name: str = None, console: Console = Console()):
        if llm_name:
            self.llm = prepare_agent(llm_name)
        else:
            self.llm = prepare_agent(get_config().get("main_pipeline.agent_logic.distillation_llm"))
        self.console = console
        self.rate_limit_delay = get_config().get('agent.rate_limit_delay', 0)

    def distill(self, content: str, query: str, thought: str, source: str = None, verbose=False) -> str:
        """Single item distillation (Legacy/Fallback)."""
        return f"Distilled Fact: Content relevant to '{query}' found."
    
    def get_result_id(self, content: str) -> str:
        """Compute a stable ID for a given content string, used to track evidence across distillation.

        Args:
            content (str): The content string for which to compute the ID.

        Returns:
            str: A stable ID string derived from the content.
        """
        return f"ev_{hash(content) & 0xFFFFFF:06x}"

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str,
                      llm_response: str = "", verbose=False) -> List[Tuple[str, List[int]]]:
        """
        Summarize multiple content items in one go.

        Args:
            items: List of (content, source_path_str) tuples.
            query: The user question.
            thought: The planner's current reasoning.
            llm_response: The main LLM's response when given the raw tool results.

        Returns:
            List of fact summary strings and their supporting evidence IDs.
        """
        pass


# ---------------------------------------------------------------------------
# Tool base classes — get_description() now returns a Mistral tool schema
# ---------------------------------------------------------------------------

class BaseGrepTool(Tool):
    """A base tool for performing grep-like operations on text data."""

    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs

    def search(self, pattern: str, source_regex: Optional[str] = None) -> List[RetrievalResult]:
        """Search for pattern in source files."""
        return []

    def shorten_results(self, pattern: str, results: List[str], limit: int) -> List[str]:
        """Shorten results to fit within limit lines."""
        return results

    def get_description(self) -> Dict[str, Any]:
        """Return a OpenAI-compatible tool schema for this tool."""
        return _tool_desc(name="grep_tool", description='', properties={})


class BaseScriptFinderTool(Tool):
    """A base tool for finding scripts in a codebase."""

    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs

    def find_scripts(self, script_names: List[str]) -> List[str]:
        """Find scripts by name. Returns a list of absolute file paths."""
        return ["/path/to/found/script.nvn"]

    def original_path(self, path: str) -> str:
        """Return the orignal path corresponding to given script"""
        return path # Placeholder

    def read_file(self, file_path: str) -> str:
        """Helper to read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Couldn't read the following file {file_path}: {e}"

    def get_description(self) -> Dict[str, Any]:
        """Return a OpenAI-compatible tool schema for this tool."""
        return _tool_desc(name="script_finder_tool", description='', properties={})



class BaseRAGTool(Tool):
    """A base tool for performing RAG operations."""

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k=None, verbose=False,
                 key_words=None, sources=None) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query."""
        return []

    def get_description(self) -> Dict[str, Any]:
        """Return a OpenAI-compatible tool schema for this tool."""
        return _tool_desc(name="rag_tool", description='', properties={})



# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    """State definition for the agent workflow sub-graph."""
    pipeline_state: AgentGraphState
    regenerate: bool
    current_thought: Optional[str]
    rewritten_prompt: Optional[str]
    local_grep_retries: Optional[Tuple[int, int]] # Contains (number of retries, last number of grep results)
    local_graph_retries: Optional[int] # Number of graph retries
    prior_evidence_end_investigation: Optional[Tuple[bool, int]]  # Contains (end_investigation, last number of retrieved evidence)
    # tool-calling round-trip state
    pending_tool_call: Optional[Dict[str, Any]]  # {tool_id, tool_name, arguments}
    continuation: Optional[bool]  # Whether this is a continuation of a previous attempt (used for planner context)


# ---------------------------------------------------------------------------
# Base workflow
# ---------------------------------------------------------------------------

class BaseAgentWorkflow(StateGraph):
    """A base workflow for agent operations."""

    def __init__(self, rag_tool: BaseRAGTool, grep_tool: BaseGrepTool,
                 script_finder_tool: BaseScriptFinderTool,
                 distillation_tool: BaseDistillationTool,
                 console: Console = Console()):
        super().__init__(WorkflowState)
        self.console = console
        self.rag_tool = rag_tool
        self.grep_tool = grep_tool
        self.script_finder_tool = script_finder_tool
        self.distillation_tool = distillation_tool
        
        with open("base_instructions.txt", "r") as file:
            self.base_instructions = file.read()

    # -----------------------------------------------------------------------
    # History helpers
    # -----------------------------------------------------------------------

    def _get_history(self, state: WorkflowState) -> List[ActionLog]:
        """Safely retrieve execution history from the pipeline state."""
        return state['pipeline_state'].get("execution_history", [])

    def _append_history(self, state: WorkflowState, tool: str, param: Any, summary: str, thought: str,
                        results_to_analyse: Optional[List[RetrievalResult]] = None) -> None:
        """Append a new action to the history."""
        history = self._get_history(state)
        new_log: ActionLog = {
            "step": len(history) + 1,
            "query": state['pipeline_state'].get("question", ""),
            "thought": thought,
            "tool": tool,
            "parameter": str(param),
            "outcome_summary": summary,
            "results_to_analyse": results_to_analyse,
        }
        if "execution_history" not in state['pipeline_state']:
            state['pipeline_state']["execution_history"] = []

        state['pipeline_state']["execution_history"].append(new_log)

    def _parse_tag(self, tag: str, text: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    # -----------------------------------------------------------------------
    # Prompt builders
    # -----------------------------------------------------------------------

    def _get_knowledge_bank_str(self, state: WorkflowState) -> str:
        """Format the knowledge bank for prompt inclusion."""
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        question = state['pipeline_state'].get("question")
        if not knowledge_bank:
            return "(No relevant facts have been gathered yet.)\n"

        elems_by_query: Dict[str, List[KnowledgeElement]] = {}
        current_elems: List[KnowledgeElement] = []
        for elem in knowledge_bank:
            query = elem.get("query", "General")
            if query == question:
                current_elems.append(elem)
                continue

            if query not in elems_by_query:
                elems_by_query[query] = []
            elems_by_query[query].append(elem)
        
        kb_str = ""
        if len(elems_by_query) > 0:
            kb_str += "## Facts related to previous queries:\n"
        
        for query, elems in elems_by_query.items():
            kb_str += f"# Query: {query}\n"
            for i, elem in enumerate(elems):
                kb_str += f"{i+1}.\n{format_knowledge_element(elem)}\n"
            kb_str += "\n"

        if len(elems_by_query) > 0:
            kb_str += f"## Facts related to the current query:\n"
            if not current_elems:
                kb_str += "(No facts have been gathered yet for the current question.)\n"

        for i, elem in enumerate(current_elems):
            kb_str += f"{i+1}.\n{format_knowledge_element(elem)}\n"

        return kb_str+"\n"

    def _show_previous_qa(self, state: WorkflowState) -> str:
        """
        Formats previous Q&A pairs from the conversation history.
        Provides context for multi-turn conversations.
        
        Returns an empty string if no previous Q&A pairs exist.
        """
        previous_qa = state['pipeline_state'].get("previous_qa", [])
        
        qa_str = "### CONVERSATION HISTORY\n"
        if not previous_qa:
            qa_str += "(No previous questions or answers.)\n"

        for i, (question, answer) in enumerate(previous_qa, 1):
            qa_str += f"\n**Previous Question {i}:**\n{question}\n\n"
            qa_str += f"**Previous Answer {i}:**\n{answer}\n"
        
        return qa_str + "\n"
    
    def _distill_results(self, state: WorkflowState, log: ActionLog) -> List[KnowledgeElement]:
        # Complete accumulated_evidence with the distilled retrieval results (key = number of elements)
        items_to_distill = []
        for result in log["results_to_analyse"]:
            items_to_distill.append((result.chunk.content, result.chunk.metadata.get('original_file_path', 'Unknown Source')))

        new_facts = self.distillation_tool.distill_batch(
            items=items_to_distill,
            query=log["query"],
            thought=log["thought"],
            previous_generation=state["pipeline_state"].get("generation", ""),
            verbose=state['pipeline_state']['verbose'],
        )
        knowledge_elements = [
            KnowledgeElement(
                fact=fact,
                tool=log["tool"],
                query=log["query"],
                evidence_ids=[self.distillation_tool.get_result_id(items_to_distill[i][0]) for i in ids]
            ) for (fact, ids) in new_facts
        ]
        state['pipeline_state'].setdefault("knowledge_bank", []).extend(knowledge_elements)
        
        accumulated_evidence = state['pipeline_state'].setdefault("accumulated_evidence", {})
        for _, indices in new_facts:
            for i in indices:
                id = self.distillation_tool.get_result_id(items_to_distill[i][0])
                if id not in accumulated_evidence:
                    accumulated_evidence[id] = log["results_to_analyse"][i]

        return knowledge_elements
    
    def _format_results(self, results: List[RetrievalResult]) -> str:
        """Formats the search results by source.

        Args:
            results (List[RetrievalResult]): A list of retrieval results, each containing content and metadata.

        Returns:
            str: A formatted string grouping results by source file with line numbers.
        """
        results_by_source = {}
        for result in results:
            source = result.chunk.metadata.get('original_file_path', 'Unknown')
            if source not in results_by_source:
                results_by_source[source] = []
            line_start, line_end = result.chunk.get_line_range()
            results_by_source[source].append({
                "content": result.chunk.content,
                "line_start": line_start,
                "line_end": line_end,
            })

        # Format output grouped by source with line numbers for each result
        raw_results_parts = []
        for source in sorted(results_by_source.keys()):
            source_results = results_by_source[source]
            # Only show count if there are multiple results from this file
            if len(source_results) == 1:
                line_start = source_results[0]["line_start"]
                line_end = source_results[0]["line_end"]
                raw_results_parts.append(f"=== Source: {source} [Lines {line_start}-{line_end}] ===")
                raw_results_parts.append(source_results[0]["content"])
                continue

            raw_results_parts.append(f"=== Source: {source} [{len(source_results)} results] ===")
            for result in source_results:
                content = result["content"]
                line_start = result["line_start"]
                line_end = result["line_end"]
                raw_results_parts.append(f"[Lines {line_start}-{line_end}]:\n{content}")

        return "\n\n".join(raw_results_parts)

    def _design_first_part_prompt(self, state: WorkflowState) -> str:
        """
        Constructs the 'World State' section that precedes every Solver prompt.
        Contains: question, conversation history, knowledge bank, current thought.
        """
        question = state['pipeline_state']['question']
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        thought = state.get('current_thought', None)

        user_prompt = ""
        
        # Add conversation history if available
        previous_qa_str = self._show_previous_qa(state)
        user_prompt += previous_qa_str + "\n"

        user_prompt += f"### CURRENT QUESTION\n{question}\n\n"
        
        if knowledge_bank:
            user_prompt += "### VERIFIED FACTS (Accumulated Knowledge)\n"
            user_prompt += self._get_knowledge_bank_str(state)
        else:
            user_prompt += "### VERIFIED FACTS\n(No relevant facts have been gathered yet.)\n\n"

        if thought:
            user_prompt += f"### CURRENT THOUGHT\n{thought}\n\n"

        return user_prompt

    def _get_optimized_history_str(self, history: List[ActionLog]) -> str:
        """Compact history string – full for ≤5 steps, compressed otherwise."""
        if not history:
            return "(No previous actions taken.)"

        total_steps = len(history)
        history_str = ""

        if total_steps <= 5:
            for log in history:
                history_str += (
                    f"- Step {log['step']}:\n"
                    f"  * Thought: {log['thought']}\n"
                    f"  * Tool: {log['tool']} -> {log['parameter']}\n"
                    f"  * Outcome: {log['outcome_summary']}\n"
                )
            return history_str

        first = history[0]
        history_str += (
            f"- Step {first['step']} (Start):\n"
            f"  * Thought: {first['thought']}\n"
            f"  * Tool: {first['tool']} -> {first['parameter']}\n"
            f"  * Outcome: {first['outcome_summary']}\n"
        )
        history_str += (
            f"\n... [Steps 2 to {total_steps - 3} were executed. Details hidden for brevity. "
            f"The agent tried various strategies which led to the current state.] ...\n\n"
        )
        for log in history[-3:]:
            history_str += (
                f"- Step {log['step']}:\n"
                f"  * Thought: {log['thought']}\n"
                f"  * Tool: {log['tool']} -> {log['parameter']}\n"
                f"  * Outcome: {log['outcome_summary']}\n"
            )
        return history_str

    def _get_anti_repetition_str(self, history: List[ActionLog], limit: int = 6) -> str:
        """Summarize recent attempts so the planner can avoid repeating itself."""
        if not history:
            return (
                "- No previous attempts yet.\n"
                "- Bootstrap rule: use the provided root tree as structural context and prefer rag_tool as the first active search when the target is still unclear."
            )

        recent = history[-limit:]
        recent_tools = [log["tool"] for log in recent]
        repetition_notes = [
            "- Do not repeat the exact same tool call with the same arguments unless new evidence justifies it.",
            "- If grep_tool already returned too many matches or zero matches, change the pattern or switch tool.",
            "- If graph_tool explored one branch repeatedly without evidence, switch to rag_tool or grep_tool with a concrete identifier.",
            "- If rag_tool results were weak, reformulate the query or extract a precise term for grep_tool.",
        ]

        if recent_tools.count("graph_tool") >= 2:
            repetition_notes.append("- graph_tool was used multiple times recently; avoid another structural-only loop unless you are exploring a genuinely new branch.")
        if recent_tools.count("grep_tool") >= 2:
            repetition_notes.append("- grep_tool was used multiple times recently; avoid retrying the same literal or source filter.")
        if recent_tools.count("rag_tool") >= 2:
            repetition_notes.append("- rag_tool was used multiple times recently; prefer a genuinely reformulated query or pivot to a structural tool.")

        for log in recent[-3:]:
            repetition_notes.append(
                f"- Recent attempt: {log['tool']} with {log['parameter']} -> {log['outcome_summary']}"
            )

        return "\n".join(repetition_notes)

    @abstractmethod
    def design_planner_prompt(self, state: WorkflowState) -> str:
        """Override in subclasses to return the planner's system-role string."""
        raise NotImplementedError("Subclasses must implement design_planner_prompt.")

    # -----------------------------------------------------------------------
    # Planner node
    # -----------------------------------------------------------------------

    @abstractmethod
    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        Planner node — must be overridden by concrete subclasses.
        Base implementation is a no-op placeholder.
        """
        raise NotImplementedError("Subclasses must implement agentic_router.")

    def decide_after_routing(self, state: WorkflowState) -> str:
        self.console.print("[dim]--- SUB-DECISION: After Routing ---[/dim]")
        if state["regenerate"]:
            if state["pipeline_state"]["verbose"]:
                self.console.print(f"[dim]    -> Routing to tool: [green]{state['pending_tool_call']['tool_name']}[/green][/dim]")
            return str(state['pending_tool_call']['tool_name'])
        return "submit_answer"

    # -----------------------------------------------------------------------
    # Tool nodes
    # -----------------------------------------------------------------------

    @abstractmethod
    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:
        """Override in subclasses to call the rag tool and update the state"""
        raise NotImplementedError("Subclasses must implement use_rag_tool.")

    @abstractmethod
    def use_grep_tool(self, state: WorkflowState) -> WorkflowState:
        """Override in subclasses to call the grep tool and update the state"""
        raise NotImplementedError("Subclasses must implement use_grep_tool.")

    @abstractmethod
    def use_script_finder_tool(self, state: WorkflowState) -> WorkflowState:
        """Override in subclasses to call the script finder tool and update the state"""
        raise NotImplementedError("Subclasses must implement use_script_finder_tool.")

    def use_simple_regeneration_tool(self, state: WorkflowState) -> WorkflowState:
        """This tool is not supposed to be called by the planner LLM directly but it is a fallback
        if the tool calling failed."""
        self.console.print("[dim]--- SUB-NODE: Simple Regeneration Tool ---[/dim]")
        self._append_history(state, "regeneration", "N/A",
                             "Refined the reasoning prompt.", state["current_thought"])
        additional_advice = state["pending_tool_call"].get("arguments", {}).get("advice", "")
        base_prompt = self._design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
            f"The previous attempt resulted in an unsatisfactory answer.\n{additional_advice}\n"
            f"Please try again, paying close attention to the previous answer."
        )
        return state

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------

    def build_graph(self):
        self.add_node("agentic_router", self.agentic_router)
        self.add_node("rag_tool", self.use_rag_tool)
        self.add_node("grep_tool", self.use_grep_tool)
        self.add_node("script_finder_tool", self.use_script_finder_tool)
        self.add_node("simple_regeneration_tool", self.use_simple_regeneration_tool)

        self.add_edge(START, "agentic_router")
        self.add_conditional_edges(
            "agentic_router",
            self.decide_after_routing,
            {
                "rag_tool": "rag_tool",
                "grep_tool": "grep_tool",
                "script_finder_tool": "script_finder_tool",
                "simple_regeneration_tool": "simple_regeneration_tool",
                "submit_answer": END,
            },
        )
        self.add_edge("rag_tool", END)
        self.add_edge("grep_tool", END)
        self.add_edge("script_finder_tool", END)
        self.add_edge("simple_regeneration_tool", END)

        return self.compile()
