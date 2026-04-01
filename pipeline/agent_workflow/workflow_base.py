import re
from abc import abstractmethod
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from langgraph.graph import END, StateGraph, START
from pipeline.langgraph_base import AgentGraphState, ActionLog
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

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str,
                      llm_response: str = "", verbose=False) -> List[str]:
        """
        Summarize multiple content items in one go.

        Args:
            items: List of (content, source_path_str) tuples.
            query: The user question.
            thought: The planner's current reasoning.
            llm_response: The main LLM's response when given the raw tool results.

        Returns:
            List of fact summary strings.
        """
        if not items:
            return []

        prompt_text = (
            f"### CONTEXT\nQuery: {query}\nCurrent Thought: {thought}\n\n"
            f"### DOCUMENTS TO ANALYZE\n"
        )
        indexed_sources = {}
        for i, (content, source) in enumerate(items):
            snippet = content[:2000]
            prompt_text += f"--- ITEM {i} (Source: {source}) ---\n{snippet}\n\n"
            indexed_sources[i] = source

        prompt_text += (
            "### INSTRUCTION\n"
            "Analyze the items above. Extract key facts that help answer the Query.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "Return the output as a list where each line is: 'Fact [Source: ITEM X]'"
        )

        distilled_results = []
        for i, source in indexed_sources.items():
            fact = f"Distilled info regarding '{query}' extracted from {source.split('/')[-1]}..."
            distilled_results.append(fact)

        return distilled_results


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
    # tool-calling round-trip state
    pending_tool_call: Optional[Dict[str, Any]]  # {tool_id, tool_name, arguments}


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

    def design_first_part_prompt(self, state: WorkflowState) -> str:
        """
        Constructs the 'World State' section that precedes every Solver prompt.
        Contains: question, knowledge bank, current thought.
        """
        question = state['pipeline_state']['question']
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        thought = state.get('current_thought', None)

        user_prompt = f"### QUESTION\n{question}\n\n"
        if knowledge_bank:
            user_prompt += "### VERIFIED FACTS (Accumulated Knowledge)\n"
            for i, fact in enumerate(knowledge_bank):
                user_prompt += f"{i+1}. {fact}\n"
            user_prompt += "\n"
        else:
            user_prompt += "### VERIFIED FACTS\n(No relevant facts have been gathered yet.)\n\n"

        if thought:
            user_prompt += f"### CURRENT THOUGHT\n{thought}\n\n"

        return user_prompt

    def _get_optimized_history_str(self, history: List["ActionLog"]) -> str:
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
        base_prompt = self.design_first_part_prompt(state)
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
