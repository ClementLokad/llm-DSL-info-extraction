"""
ConcreteAgentWorkflow
=====================
Implements the Planner using the Mistral native tool-calling API.

Key changes vs the XML-based version
--------------------------------------
* ``agentic_router`` calls ``planner_llm.generate_with_tools(...)``
  – the model returns a guaranteed-JSON tool_call block instead of XML.
* All embedded XML parameter tags (<sources>, <block_type>, <key_words>)
  are gone.  Each tool now declares those fields as first-class JSON
  properties in its ``get_description()`` schema, and the tool nodes
  read them from ``state["pending_tool_call"]["arguments"]``.
* Invalid-tool fallback is preserved: if the model somehow returns an
  unknown name (shouldn't happen with tool_choice="any"), we redirect
  to simple_regeneration_tool exactly as before.
* The grep-refinement follow-up loop uses ``planner_llm.submit_tool_result``
  so the history stays coherent (tool → assistant roles).
* Tree-tool results are submitted via ``submit_tool_result`` as well.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import config_manager
from pipeline.agent_workflow.workflow_base import (
    Tool,
    BaseAgentWorkflow,
    BaseDistillationTool,
    BaseGrepTool,
    BaseRAGTool,
    BaseScriptFinderTool,
    WorkflowState,
    _tool_desc,
)
from agents.prepare_agent import prepare_agent
from agents.base import ToolCallResult
from rag.core.base_parser import BlockType
from pipeline.agent_workflow.rag_tool import AdvancedRAGTool
from config_manager import get_config
from langgraph.graph import END, StateGraph, START

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.markup import escape


# ---------------------------------------------------------------------------
# Tree tool
# ---------------------------------------------------------------------------

class BaseTreeTool(Tool):
    def tree_tool(self, root_path: str, max_tokens: int = 1000) -> str:
        return ""

    def custom_tree(self, root_path: str, max_depth: int, max_children: int) -> str:
        return ""

    def get_description(self) -> Dict[str, Any]:
        """Return a OpenAI-compatible tool schema for this tool."""
        return _tool_desc(name="tree_tool", description='', properties={})


# ---------------------------------------------------------------------------
# Planner tool schema — the single 'virtual' tool the planner selects from
# ---------------------------------------------------------------------------

def _build_planner_tools(tools: List[Tool], include_grade: bool = True) -> List[Dict[str, Any]]:
    """
    Return the list of tool schemas passed to the Mistral planner call.

    Each real retrieval tool is declared with its own schema so the model
    directly fills in typed, validated arguments.  ``grade_answer`` is a
    zero-argument signal to stop the loop.
    """
    tools = [tool.get_description() for tool in tools]
    if include_grade:
        tools.append(
            _tool_desc(
                name="grade_answer",
                description=(
                    "This tool stops the investigation loop and initiates grading phase."
                ),
                properties={},
                required=[],
            )
        )
    return tools


# ---------------------------------------------------------------------------
# ConcreteAgentWorkflow
# ---------------------------------------------------------------------------

VALID_TOOLS = {
    "rag_tool",
    "grep_tool",
    "script_finder_tool",
    "tree_tool",
    "grade_answer",
}


class ConcreteAgentWorkflow(BaseAgentWorkflow):
    """
    Concrete agentic workflow backed by Mistral native tool-calling.
    """

    def __init__(
        self,
        rag_tool: BaseRAGTool,
        grep_tool: BaseGrepTool,
        script_finder_tool: BaseScriptFinderTool,
        distillation_tool: BaseDistillationTool,
        tree_tool: BaseTreeTool,
        console: Console = Console(),
    ):
        super().__init__(rag_tool, grep_tool, script_finder_tool,
                         distillation_tool, console)
        self.tree_tool = tree_tool
        self.config_manager = config_manager.get_config()
        self.planner_llm = prepare_agent(
            self.config_manager.get(
                'main_pipeline.agent_logic.planner_llm',
                self.config_manager.get_default_agent(),
            )
        )
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)

    # -----------------------------------------------------------------------
    # Planner system prompts
    # -----------------------------------------------------------------------

    def _kickoff_system_prompt(self, state: WorkflowState) -> str:
        question = state['pipeline_state']['question']
        tree_str = self.tree_tool.custom_tree('', 3, 3)
        return (
            self.base_instructions
            + "\n### SYSTEM ROLE\n"
            "You are the **Strategic Planner** for an advanced RAG agent working on a "
            "**Lokad Envision** codebase.\n"
            "Lokad is a supply chain optimization company. Envision is their specialized "
            "programming language for quantitative supply chain logic.\n"
            "You are initiating a new investigation. Select the best FIRST tool to gather information.\n\n"
            f"### MISSION GOAL\n{question}\n\n"
            "### CODEBASE STRUCTURE (top 3 levels)\n"
            f"{tree_str}\n\n"
            "### PLANNING INSTRUCTIONS\n"
            "1. Identify the most critical keyword or concept in the Mission Goal.\n"
            "2. Choose the tool that matches the nature of what you are looking for:\n"
            " - The question mentions a specific file path, function name, variable,"
            " or string literal that is unlikely to appear in unrelated contexts\n"
            "    → grep_tool is precise and fast for exact matches.\n"
            " - The question asks about a concept, a behaviour, business logic,"
            " or uses broad terms that could appear in many different contexts\n"
            "    → rag_tool finds semantically relevant chunks regardless of exact wording.\n"
            " - When genuinely uncertain, consider what a first result would look like: "
            "if a grep would likely return hundreds of unrelated matches, use rag_tool.\n"
            "3. Call exactly ONE tool. Fill in all RELEVANT optional fields for precision.\n"
        )

    def _continuation_system_prompt(self, state: WorkflowState) -> str:
        question = state['pipeline_state']['question']
        history = self._get_history(state)
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        previous_generation = state['pipeline_state'].get('generation') or "(No generation yet)"

        facts_str = ""
        if knowledge_bank:
            facts_str = "\n".join(f"{i}. {f}" for i, f in enumerate(knowledge_bank))
        else:
            facts_str = "(Knowledge bank is empty.)"

        return (
            self.base_instructions
            + "\n### SYSTEM ROLE\n"
            "You are the **Investigation Supervisor** for an advanced RAG agent on a "
            "**Lokad Envision** codebase.\n"
            "Your primary goal is EFFICIENCY. Decide whether current information is "
            "sufficient to answer the question.\n"
            "If the answer is found, call grade_answer immediately.\n\n"
            f"### MISSION GOAL\n{question}\n\n"
            "### PROPOSED SOLUTION (From Main Agent)\n"
            f"\"{previous_generation}\"\n\n"
            "**CRITICAL CHECK**: Does the proposed solution directly and fully answer the "
            "Mission Goal? If YES, call grade_answer. "
            "Vague answers or 'I don't know' should be treated as NOT ANSWERED.\n\n"
            f"### VERIFIED FACTS\n{facts_str}\n\n"
            "### HISTORY (Previous Steps)\n"
            f"{self._get_optimized_history_str(history)}\n\n"
            "### DECISION LOGIC\n\n"
            "1. COMPLETION CHECK: Answer these two questions independently:\n"
            "a. Does the Proposed Solution address the Mission Goal? (form)\n"
            "b. Is it grounded in specific facts retrieved from the codebase"
            " with at least one concrete source cited? (evidence)\n"
            "Only call grade_answer if BOTH are YES.\n"
            "A negative conclusion (\"this is not done\", \"I found nothing\") answers (a) "
            "but FAILS (b) unless search results explicitly confirmed the absence. "
            "In that case, proceed to the Exhaustiveness Check.\n\n"
            "2. EXHAUSTIVENESS CHECK: If the answer is negative or uncertain, ask yourself:\n"
            "- Did I try both rag_tool AND grep_tool?\n"
            "- Did I try boosting rag results with sources and/or kewords ?\n"
            "- Did I vary the query language (e.g. synonyms, related concepts)?\n"
            "If any answer is NO → proceed to step 3.\n\n"
            "3. GAP ANALYSIS: What specific information is missing? Choose the most targeted tool.\n"
            "Check History: if grep returned 0 or too many results, switch to rag_tool — "
            "semantic search handles broad concepts far better than exact matching. "
            "If rag_tool returned weak results, then try grep with a more specific identifier "
            "extracted from those results.\n"
        )

    # -----------------------------------------------------------------------
    # Planner node  (uses Mistral tool-calling)
    # -----------------------------------------------------------------------
    # TODO: Favorize rag tool more

    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        Planner node.

        Calls the tool-calling API.  The model is forced to call one
        of the declared tools (tool_choice="any"), so we always get a
        well-structured JSON response — no XML parsing, no regex.
        Note that some API's like Ollama do not guarantee that a valid
        tool is called -> a ValueError is raised in this case.

        Grep refinement and tree-tool follow-ups use submit_tool_result so
        that the conversation history stays coherent (tool → assistant roles).
        """
        self.console.print("[dim]--- SUB-NODE: Agentic Router (Planner) ---[/dim]")

        history = self._get_history(state)
        already_distilled: bool = False
        reasoning = None
        is_continuation = bool(history)
        include_grade = is_continuation  # grade_answer only available after first step

        # Planner tools available at this turn
        planner_tools = _build_planner_tools(tools=[
            self.rag_tool,
            self.grep_tool,
            self.tree_tool,
            self.script_finder_tool 
        ], include_grade=include_grade)

        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        
        feedback = ""

        # ------------------------------------------------------------------
        # Grep-refinement branch
        # Reuses the open conversation via submit_tool_result so the model
        # sees: user → assistant[tool_calls] → tool[result] → assistant ...
        # ------------------------------------------------------------------
        if (
            "local_grep_retries" in state
            and state["pipeline_state"].get("execution_history")
            and state["pipeline_state"]["execution_history"][-1]["tool"] == "grep_tool"
        ):
            already_distilled = True
            len_res = state["local_grep_retries"][1]
            # Remove the tentative history entry so it doesn't double-count
            state["pipeline_state"]["execution_history"] = \
                state["pipeline_state"]["execution_history"][:-1]
            
            pending = state.get("pending_tool_call", {})
            
            feedback += "\n".join(f"Warning: {w}" for w in pending.get("arguments", {}).get("warnings", [])) + "\n"

            if len_res > get_config().get("main_pipeline.grep_tool.max_results_to_refine"):
                feedback += (
                    f"The grep search yielded {len_res} results, which exceeds the "
                    f"limit of {get_config().get('main_pipeline.grep_tool.max_results_to_refine')}. "
                )
                instructions = (
                    "Please narrow the search pattern or switch to a different tool. "
                    "Respond by calling one of the available tools."
                )
            else:
                feedback += "The grep search returned 0 results. "
                instructions = (
                    "Please broaden the search pattern or switch to a different tool. "
                    "Respond by calling one of the available tools."
                )
            try:
                tool_call = self.planner_llm.submit_tool_result_and_continue(
                    tool_call_id=pending.get("tool_id", "unknown"),
                    tool_name="grep_tool",
                    result=feedback,
                    next_instruction=instructions,
                    tools=planner_tools,
                    tool_choice="any",
                )
            except ValueError as exc:
                # Model fell back to plain-text output — redirect to regeneration
                self.console.print(f"[dim][yellow]Planner fallback (grep refine): {exc}[/yellow][/dim]")
                tool_call = ToolCallResult(
                    tool_name="simple_regeneration_tool",
                    tool_id="fallback",
                    arguments={"advice": str(exc)},
                )

        # ------------------------------------------------------------------
        # Tree-tool follow-up branch
        # ------------------------------------------------------------------
        elif (
            history
            and history[-1]["tool"] == "tree_tool"
            and state.get("rewritten_prompt")
        ):
            already_distilled = True
            tree_str = state["rewritten_prompt"]
            state["pipeline_state"]["execution_history"] = \
                state["pipeline_state"]["execution_history"][:-1]

            pending = state.get("pending_tool_call", {})
            
            feedback += (
                "Here is the directory tree (rooted at the longest valid prefix "
                f"of the path you provided):\n\n{tree_str}."
                "\nBased on the tree above, select the next tool to call."
            )
            
            try:
                tool_call = self.planner_llm.submit_tool_result_and_continue(
                    tool_call_id=pending.get("tool_id", "unknown"),
                    tool_name="tree_tool",
                    result=(
                        "Here is the directory tree (rooted at the longest valid prefix "
                        f"of the path you provided):\n\n{tree_str}"
                    ),
                    next_instruction="Based on the tree above, select the next tool to call.",
                    tools=planner_tools,
                    tool_choice="any",
                )
            except ValueError as exc:
                self.console.print(f"[dim][yellow]Planner fallback (tree follow-up): {exc}[/yellow][/dim]")
                tool_call = ToolCallResult(
                    tool_name="simple_regeneration_tool",
                    tool_id="fallback",
                    arguments={"advice": str(exc)},
                )

        # ------------------------------------------------------------------
        # Normal planning branch
        # ------------------------------------------------------------------
        else:
            if is_continuation:
                system_prompt = self._continuation_system_prompt(state)

                # Step 1: let the planner reason freely in text before committing to a tool.
                # This forces it to execute the flowchart explicitly rather than
                # pattern-matching to a tool in a single cold inference.
                self.planner_llm.reset_context()
                reasoning = self.planner_llm.generate_response(
                    user_message=(
                        "Follow the flowchart in your instructions step by step. "
                        "Write out your reasoning for each step explicitly, "
                        "then state which tool you will call and why."
                    ),
                    system_prompt=system_prompt,
                    temperature=0.2,
                )
                # Notice that reasoning is not used explicitely, it is only to help the model
                # thinking correctly

                # Step 2: now force a tool call, with the reasoning already in context.
                # The model sees its own analysis and must act consistently with it.
                try:
                    tool_call = self.planner_llm.generate_with_tools(
                        user_message="Based on your reasoning above, call the appropriate tool now.",
                        tools=planner_tools,
                        tool_choice="any",
                        # No system_prompt here — it is already in self.context from step 1
                    )
                except ValueError as exc:
                    self.console.print(f"[dim][yellow]Planner fallback (normal): {exc}[/yellow][/dim]")
                    tool_call = ToolCallResult(
                        tool_name="simple_regeneration_tool",
                        tool_id="fallback",
                        arguments={"advice": (
                            "The planner failed to select a valid tool. "
                            "Please re-examine the question and select the most appropriate tool."
                        )},
                    )

            else:
                system_prompt = self._kickoff_system_prompt(state)
                # Kickoff: single call is fine, no reasoning needed
                self.planner_llm.reset_context()
                try:
                    tool_call = self.planner_llm.generate_with_tools(
                        user_message="Begin the investigation. Select the best first tool to use.",
                        tools=planner_tools,
                        system_prompt=system_prompt,
                        tool_choice="any",
                    )
                except ValueError as exc:
                    self.console.print(f"[dim][yellow]Planner fallback (normal): {exc}[/yellow][/dim]")
                    tool_call = ToolCallResult(
                        tool_name="simple_regeneration_tool",
                        tool_id="fallback",
                        arguments={"advice": (
                            "The planner failed to select a valid tool. "
                            "Please re-examine the question and select the most appropriate tool."
                        )},
                    )

        # ------------------------------------------------------------------
        # Extract and validate the tool decision
        # ------------------------------------------------------------------
        chosen_tool = tool_call.tool_name
        arguments = tool_call.arguments
        tool_id = tool_call.tool_id

        # Build a human-readable parameter summary for logs / history
        param_summary = json.dumps(arguments, ensure_ascii=False)

        # Extract "thought" from arguments if the model put one there,
        # otherwise synthesise a short description for the history log.
        thought = arguments.pop("thought", f"Calling {chosen_tool}.")

        if chosen_tool not in VALID_TOOLS:
            # Self-correction: redirect to regeneration
            self.console.print(
                f"  [System Warning]: Unknown tool '{chosen_tool}' → redirecting."
            )
            chosen_tool = "simple_regeneration_tool"
            arguments = {
                "advice": (
                    f"SYSTEM ERROR: The tool '{tool_call.tool_name}' does not exist. "
                    f"Valid tools: {', '.join(VALID_TOOLS)}."
                )
            }
            tool_id = tool_call.tool_id

        state['current_thought'] = thought
        state['pending_tool_call'] = {
            "tool_id": tool_id,
            "tool_name": chosen_tool,
            "arguments": arguments,
        }

        max_retries = self.config_manager.get("main_pipeline.agent_logic.max_retries", 5)
        state["regenerate"] = (
            chosen_tool != "grade_answer"
            and state["pipeline_state"]["retry_count"] <= max_retries
        )

        if state["pipeline_state"]["verbose"]:
            if feedback != "": # There was a follow-up prompt
                prompt_content = Text.from_markup(f"Follow-up prompt: {escape(feedback)}\n")
            else:
                prompt_content = Panel(escape(system_prompt), title="Planner Prompt", border_style="purple")
            
            if reasoning:
                reasoning_content = Panel(
                    Markdown(reasoning),
                    title="Planner Reasoning",
                    border_style="yellow",
                )

            tool_content = Text.from_markup(
                f"\nPlanner selected tool: [bold green]{chosen_tool}[/bold green] "
                f"with arguments: [bold orange3]{escape(param_summary)}[/bold orange3]\n"
            )
            thought_content = Panel(Markdown(thought), title="Planner Thought", border_style="blue")
            if reasoning:
                self.console.print(Panel(Group(prompt_content, reasoning_content, tool_content, thought_content),
                                         title=f"Planner - {self.planner_llm.model_name}", border_style="bright_red"))
            else:
                self.console.print(Panel(Group(prompt_content, tool_content, thought_content),
                                         title=f"Planner - {self.planner_llm.model_name}", border_style="bright_red"))

        # ------------------------------------------------------------------
        # Distil previous tool results (lazy distillation)
        # ------------------------------------------------------------------
        exec_history = state["pipeline_state"].get("execution_history", [])
        if exec_history and not already_distilled:
            prev_results = exec_history[-1].get("results_to_analyse")
            if prev_results and chosen_tool != "grade_answer":
                self.console.print("[dim]Distilling previous tool results...[/dim]")
                items_to_distill = [
                    (r.chunk.content,
                     r.chunk.metadata.get('original_file_path', 'Unknown Source'))
                    for r in prev_results
                ]
                new_facts = self.distillation_tool.distill_batch(
                    items=items_to_distill,
                    query=state['pipeline_state']["question"],
                    thought=exec_history[-1]["thought"],
                    previous_generation=state["pipeline_state"].get("generation", ""),
                    verbose=state['pipeline_state']['verbose'],
                )
                state['pipeline_state'].setdefault("knowledge_bank", []).extend(new_facts)
                state["pipeline_state"]["execution_history"][-1]["outcome_summary"] += (
                    f" Extracted {len(new_facts)} relevant facts."
                )
                if state["pipeline_state"]["verbose"]:
                    self.console.print(f"[dim]Extracted {len(new_facts)} facts.[/dim]")

        return state

    # -----------------------------------------------------------------------
    # Tool nodes  (read clean args from pending_tool_call)
    # -----------------------------------------------------------------------

    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:
        """Semantic search – all parameters come from JSON, no regex needed."""
        self.console.print("[dim]--- SUB-NODE: RAG Tool ---[/dim]")
        args = state.get("pending_tool_call", {}).get("arguments", {})
        query = args.get("query", "")
        key_words: Optional[List[str]] = args.get("key_words") or None
        sources = args.get("sources") or None
        thought = state.get("current_thought", "No reasoning provided.")
        
        if key_words and state['pipeline_state']['verbose']:
            self.console.print(f"[dim]Key words detected in query: [bright_magenta]{escape(str(key_words))}[/bright_magenta][/dim]")

        if sources and state['pipeline_state']['verbose']:
            self.console.print(f"[dim]Sources detected in query: [sea_green1]{escape(str(sources))}[/sea_green1][/dim]")

        results = self.rag_tool.retrieve(
            query=query,
            verbose=state['pipeline_state']['verbose'],
            key_words=key_words,
            sources=sources,
        )

        if state['pipeline_state']['verbose']:
            self.console.print(
                f"RAG retrieved {len(results)} results for query: '{escape(query)}'"
            )

        count = len(results)
        outcome_str = f"Retrieved {count} chunks."
        self._append_history(state, "rag_tool", query, outcome_str, thought, results)

        raw_results_str = "\n\n".join(
            f"=== Source: {r.chunk.metadata.get('original_file_path', 'Unknown')} ===\n"
            f"{r.chunk.content}"
            for r in results
        )

        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### RAG RESULTS\n"
            f"The RAG tool retrieved {count} relevant chunks for the query '{query}':\n"
            f"{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            "Using the RAG results above and all previous knowledge, answer the question "
            "as concisely as possible."
        )
        return state

    def use_grep_tool(self, state: WorkflowState) -> WorkflowState:
        """Text grep – all parameters come from JSON, no regex needed."""
        self.console.print("[dim]--- SUB-NODE: Grep Tool ---[/dim]")
        args: Dict = state.get("pending_tool_call", {}).get("arguments", {})
        args.setdefault("warnings", [])
        pattern: str = args.get("pattern", "")
        source_regex: Optional[str] = args.get("sources") or None

        # block_type: convert comma-separated string to BlockType enum list
        block_type_raw: Optional[List[BlockType]] = args.get("block_type") or None
        block_types: Optional[List[BlockType]] = None
        if block_type_raw:
            block_types = []
            invalid_bt = []
            for bt in block_type_raw:
                bt_clean = bt.strip().upper()
                try:
                    block_types.append(BlockType[bt_clean])
                except KeyError:
                    invalid_bt.append(bt_clean)
            if not block_types:
                args["warnings"].append(f"All of the given block types are invalid so block types filter was ignored.")
                block_types = None
            elif invalid_bt:
                args["warnings"].append(f"The following given block types are invalid: {', '.join(invalid_bt)}")

        thought = state.get("current_thought", "No reasoning provided.")

        # ---- execute ----
        results = self.grep_tool.search(
            pattern=pattern, source_regex=source_regex, bloc_type=block_types, warnings=args["warnings"]
        )

        if "local_grep_retries" not in state:
            state["local_grep_retries"] = (0, len(results))
        else:
            state["local_grep_retries"] = (state["local_grep_retries"][0] + 1, len(results))

        if state['pipeline_state']['verbose']:
            self.console.print(f"Grep found {len(results)} matches for pattern: '{escape(pattern)}'")

        shortened_res = False
        original_result_count = len(results)
        max_res = get_config().get("main_pipeline.grep_tool.max_results")
        if len(results) > max_res:
            results = results[:max_res]
            shortened_res = True

        if state['pipeline_state']['verbose'] and shortened_res:
            self.console.print(f"Only the first {max_res} will be analysed")

        max_grep_retries = self.config_manager.get("main_pipeline.grep_tool.max_grep_retries", 3)
        if not results and state["local_grep_retries"][0] >= max_grep_retries:
            state['pipeline_state'].setdefault("knowledge_bank", []).append(
                f"No matches found for pattern '{pattern}'"
                + (f" in sources matching '{source_regex}'." if source_regex else ".")
            )
        
        total_sources = {r.chunk.metadata.get('original_file_path', 'Unknown') for r in results}

        outcome_str = f"Grep found {original_result_count} matches from {len(total_sources)} distinct scripts."
        if shortened_res:
            outcome_str += f" Only the first {max_res} were analyzed."
        self._append_history(state, "grep_tool", pattern, outcome_str, thought, results)

        max_lines = self.config_manager.get("main_pipeline.grep_tool.max_lines")
        compacted = self.grep_tool.shorten_results(
            pattern, [r.chunk.content for r in results], max_lines
        )
        if len(compacted) < len(results):
            raise Exception("Lost results during compaction")

        raw_results_str = "\n\n".join(
            f"=== Source: {results[i].chunk.metadata.get('original_file_path', 'Unknown')} ===\n"
            f"{compacted[i]}"
            for i in range(len(results))
        )

        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = f"{base_prompt}### GREP RESULTS\n"

        if shortened_res:
            state['rewritten_prompt'] += (
                f"WARNING: Results truncated; only the first {max_res} out of "
                f"{original_result_count} are shown.\n"
            )
        elif not results:
            state['rewritten_prompt'] += (
                f"WARNING: No matches for pattern '{pattern}'"
                + (f" in sources matching '{source_regex}'.\n" if source_regex else ".\n")
            )
        else:
            source_warn = "No files in the codebase match the source_regex so source filter was ignored."
            if source_regex and (not source_warn in args["warnings"]):
                state['rewritten_prompt'] += (
                    "The search was restricted to code blocks from files whose path match "
                    f"this regex: {source_regex}\n"
                )
            state['rewritten_prompt'] += (
                f"The Grep tool found {len(results)} matches for '{pattern}' "
                f"from {len(total_sources)} distinct scripts."
            )

        state['rewritten_prompt'] += (
            f"\n\n{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            "Using the grep results above and all previous knowledge, answer the "
            "question as concisely as possible."
        )
        return state
    
    def use_script_finder_tool(self, state: WorkflowState) -> WorkflowState:
        self.console.print("[dim]--- SUB-NODE: Script Finder Tool ---[/dim]")
        args = state.get("pending_tool_call", {}).get("arguments", {})
        script_names = args.get("script_names") or []

        thought = state.get("current_thought", "No reasoning provided.")

        found_paths = self.script_finder_tool.find_scripts(script_names=script_names)

        new_facts = []
        for path in found_paths:
            file_content = self.script_finder_tool.read_file(path)
            summary = self.distillation_tool.distill(
                content=file_content,
                query=state["pipeline_state"]["question"],
                thought=thought,
                source=self.script_finder_tool.original_path(path),
                verbose=state['pipeline_state']['verbose']
            )
            if summary.strip().upper() != "IRRELEVANT":
                new_facts.append(summary)

        state['pipeline_state'].setdefault("knowledge_bank", []).extend(new_facts)

        outcome_str = f"Found {len(found_paths)} scripts. Content read and distilled."
        if len(found_paths) < len(script_names):
            outcome_str += (
                f" {len(script_names) - len(found_paths)} could not be found."
                f" Only found: {found_paths}"
            )
        self._append_history(state, "script_finder_tool", script_names, outcome_str, thought)

        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
        )
        if found_paths:
            state['rewritten_prompt'] += (
                f"The scripts {found_paths} have been located and analyzed, "
                f"the details have been added to Verified Facts.\n"
            )
        else:
            state['rewritten_prompt'] += (
                f"Unfortunately, none of the required scripts could be found."
            )
        state['rewritten_prompt'] += (
            f"If additional information is needed, specify what is missing. "
            f"Otherwise, analyze the results to answer the question.\n"
        )
        return state

    def use_tree_tool(self, state: WorkflowState) -> WorkflowState:
        self.console.print("[dim]--- SUB-NODE: Tree Tool ---[/dim]")
        args = state.get("pending_tool_call", {}).get("arguments", {})
        root_path: str = args.get("root_path", "").strip().strip("'\"")

        tree: str = self.tree_tool.tree_tool(
            root_path, self.config_manager.get("main_pipeline.file_tree_max_tok")
        )
        real_root = "Database Root" if tree.startswith("├") else tree.split("\n")[0]

        # Store tree in rewritten_prompt; agentic_router picks it up next turn
        state["rewritten_prompt"] = tree
        outcome_str = f"Successfully generated file tree from root: {real_root}"
        self._append_history(state, "tree_tool", root_path, outcome_str,
                             state.get("current_thought", ""))
        return state

    # -----------------------------------------------------------------------
    # Grep-refinement conditional edge
    # -----------------------------------------------------------------------

    def refine_grep(self, state: WorkflowState) -> str:
        self.console.print("[dim]--- SUB-DECISION: Grep results validation ---[/dim]")
        num_retries, len_grep_results = state["local_grep_retries"]
        max_retries = self.config_manager.get("main_pipeline.grep_tool.max_grep_retries", 3)

        if num_retries >= max_retries:
            self.console.print("[dim]    -> Max grep retries reached. Validated.[/dim]")
            return "validated"

        if len_grep_results == 0:
            self.console.print("[dim]    -> 0 results, replanning.[/dim]")
            return "replan"

        if len_grep_results > self.config_manager.get("main_pipeline.grep_tool.max_results_to_refine"):
            self.console.print(
                f"[dim]    -> {len_grep_results} results > limit = {get_config().get('main_pipeline.grep_tool.max_results_to_refine')}, replanning.[/dim]"
            )
            return "replan"

        self.console.print("[dim]    -> Grep results validated.[/dim]")
        return "validated"

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------

    def build_graph(self):
        self.add_node("agentic_router", self.agentic_router)
        self.add_node("rag_tool", self.use_rag_tool)
        self.add_node("grep_tool", self.use_grep_tool)
        self.add_node("tree_tool", self.use_tree_tool)
        self.add_node("script_finder_tool", self.use_script_finder_tool)
        self.add_node("simple_regeneration_tool", self.use_simple_regeneration_tool)

        self.add_edge(START, "agentic_router")
        self.add_conditional_edges(
            "agentic_router",
            self.decide_after_routing,
            {
                "rag_tool": "rag_tool",
                "grep_tool": "grep_tool",
                "tree_tool": "tree_tool",
                "script_finder_tool": "script_finder_tool",
                "simple_regeneration_tool": "simple_regeneration_tool",
                "grade_answer": END,
            },
        )
        self.add_conditional_edges(
            "grep_tool",
            self.refine_grep,
            {"replan": "agentic_router", "validated": END},
        )
        self.add_edge("tree_tool", "agentic_router")
        self.add_edge("rag_tool", END)
        self.add_edge("script_finder_tool", END)
        self.add_edge("simple_regeneration_tool", END)

        return self.compile()
