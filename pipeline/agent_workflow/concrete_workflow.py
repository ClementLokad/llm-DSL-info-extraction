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
import os
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
from get_mapping import get_file_mapping
from pipeline.langgraph_base import KnowledgeElement, RetrievalResult, ActionLog
from agents.prepare_agent import prepare_agent
from agents.base import ToolCallResult
from rag.core.base_parser import BlockType
from pipeline.agent_workflow.graph_tool import EnvisionGraphTool
from pipeline.agent_workflow.prior_evidence_tool import PriorEvidenceTool
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
    directly fills in typed, validated arguments.  ``submit_answer`` is a
    zero-argument signal to stop the loop.
    """
    tools = [tool.get_description() for tool in tools]
    if include_grade:
        tools.append(
            _tool_desc(
                name="submit_answer",
                description=(
                    "This tool stops the investigation loop and submits the final answer."
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
    "graph_tool",
    "script_finder_tool",
    "tree_tool",
    "prior_evidence_tool",
    "simple_regeneration_tool",
    "submit_answer",
}


class ConcreteAgentWorkflow(BaseAgentWorkflow):
    """
    Concrete agentic workflow backed by Mistral native tool-calling.
    """

    def __init__(
        self,
        rag_tool: BaseRAGTool,
        grep_tool: BaseGrepTool,
        graph_tool: EnvisionGraphTool,
        script_finder_tool: BaseScriptFinderTool,
        distillation_tool: BaseDistillationTool,
        tree_tool: BaseTreeTool,
        prior_evidence_tool: PriorEvidenceTool = None,
        console: Console = Console(),
    ):
        super().__init__(rag_tool, grep_tool, script_finder_tool,
                         distillation_tool, console)
        self.tree_tool = tree_tool
        self.graph_tool = graph_tool
        self.prior_evidence_tool = prior_evidence_tool or PriorEvidenceTool()
        self.mapping = get_file_mapping()
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
        kickoff_tree_max_depth = self.config_manager.get(
            "main_pipeline.kickoff_tree_max_depth", 5
        )
        kickoff_tree_max_children = self.config_manager.get(
            "main_pipeline.kickoff_tree_max_children", 20
        )
        tree_str = self.tree_tool.custom_tree(
            '',
            kickoff_tree_max_depth,
            kickoff_tree_max_children,
        )
        facts_str = ""
        if state.get("pipeline_state", {}).get("knowledge_bank"):
            facts_str = f"### VERIFIED FACTS (from previous queries)\n{self._get_knowledge_bank_str(state)}"
        
        previous_qa_str = self._show_previous_qa(state)
        
        return (
            self.base_instructions
            + "\n### SYSTEM ROLE\n"
            "You are the **Strategic Planner** for an advanced RAG agent working on a "
            "**Lokad Envision** codebase.\n"
            "Lokad is a supply chain optimization company. Envision is their specialized "
            "programming language for quantitative supply chain logic.\n"
            "You are initiating a new investigation. Select the best FIRST tool to gather information.\n\n"
            f"{previous_qa_str}"
            f"### CURRENT QUESTION\n{question}\n\n"
            "### CODEBASE STRUCTURE (from root : '/')\n"
            f"{tree_str}\n\n"
            f"{facts_str}"
            "### PLANNING INSTRUCTIONS\n"
            "1. Identify the most critical keyword or concept in the current question.\n"
            "2. Choose the tool that matches the nature of what you are looking for:\n"
            " - The question mentions a specific file path, function name, variable,"
            " or string literal that is unlikely to appear in unrelated contexts\n"
            "    → grep_tool is precise and fast for exact matches.\n"
            " - The question asks about a concept, a behaviour, business logic,"
            " or uses broad terms that could appear in many different contexts\n"
            "    → rag_tool finds semantically relevant chunks regardless of exact wording.\n"
            " - For grep_tool sources, ONLY use real codebase path patterns (from mapping/tree).\n"
            "   Never use memory/context headings such as ENVISION Hot Memory, LIVE SESSION, or RECENT TURNS as sources.\n"
            " - When genuinely uncertain, consider what a first result would look like: "
            "if a grep would likely return hundreds of unrelated matches, use rag_tool.\n"
            "3. Call exactly ONE tool. Fill in all RELEVANT optional fields for precision.\n"
        )

    def _continuation_system_prompt(self, state: WorkflowState) -> str:
        question = state['pipeline_state']['question']
        history = self._get_history(state)
        previous_generation = state['pipeline_state'].get('generation') or "(No generation yet)"

        facts_str = self._get_knowledge_bank_str(state)
        previous_qa_str = self._show_previous_qa(state)

        tree_str = self.tree_tool.custom_tree('', 3, 3)
        return (
            self.base_instructions
            + "\n### SYSTEM ROLE\n"
            "You are the **Investigation Supervisor** for an advanced RAG agent on a "
            "**Lokad Envision** codebase.\n"
            "Your primary goal is EFFICIENCY. Decide whether current information is "
            "sufficient to answer the question.\n"
            "If the answer is found, call submit_answer immediately.\n\n"
            f"{previous_qa_str}"
            f"### CURRENT QUESTION\n{question}\n\n"
            "### CODEBASE STRUCTURE (from root : '/')\n"
            f"{tree_str}\n\n"
            "### PROPOSED SOLUTION (From Main Agent)\n"
            f"\"{previous_generation}\"\n\n"
            "**CRITICAL CHECK**: Does the proposed solution directly and fully answer the "
            "Current Question? If YES, call submit_answer. "
            "Vague answers or 'I don't know' should be treated as NOT ANSWERED.\n\n"
            f"### VERIFIED FACTS\n{facts_str}\n\n"
            "### HISTORY (Previous Steps for the current question)\n"
            f"{self._get_optimized_history_str(history)}\n\n"
            "### DECISION LOGIC\n\n"
            "Read the Proposed Solution. Ask yourself one question:\n"
            "**Does it directly and specifically answer the Current Question, "
            "using at least one concrete piece of evidence (a file path, a property or a tool call from the HISTORY)?**\n\n"
            "→ YES: call submit_answer immediately. Do not search further.\n"
            "→ NO (vague, uncertain, 'I don't know', or negative without proof): "
            "search for what is missing.\n\n"
            "The proposed solution does not need to explicitly *cite* evidence as long as it aligns with results "
            "from the HISTORY or facts from VERIFIED FACTS. The answer must still be grounded in concrete evidence.\n"
            "If you need to search further, here are useful angles to consider:\n"
            "- If the answer is negative or uncertain, have you tried both rag_tool and grep_tool?\n"
            "- If grep returned too many or 0 results, try rag_tool with the core concept instead.\n"
            "- If rag results were weak, try grep with a specific identifier extracted from those results.\n"
            "- Synonyms or rephrased queries often surface what a direct query missed.\n"
            "- prior_evidence_tool can combine documents from earlier searches without re-running them.\n"
        )

    # -----------------------------------------------------------------------
    # Planner node  (uses Mistral tool-calling)
    # -----------------------------------------------------------------------
    
    def _plan_grep_refinement(self, state, planner_tools) -> ToolCallResult:
        """
        Handle the grep-refinement branch.
    
        Called when the previous tool was grep_tool and its result count fell
        outside the acceptable range (0 or > max_results_to_refine).  We reuse
        the open planner conversation via submit_tool_result_and_continue so the
        model sees a coherent:
            user → assistant[tool_calls: grep] → tool[feedback] → assistant[tool_calls: ?]
        sequence rather than a cold re-start.
    
        Side-effect: removes the last (tentative) history entry to avoid
        double-counting it once the refined search is recorded.
        """
        # --- Step 1: Remove the tentative grep history entry ---
        # The entry was appended optimistically by use_grep_tool; if we are here
        # the result was unsatisfactory so we discard it and let the refined call
        # produce a fresh entry.
        state["pipeline_state"]["execution_history"] = (
            state["pipeline_state"]["execution_history"][:-1]
        )
    
        # --- Step 2: Build feedback message from warnings + result count ---
        pending = state.get("pending_tool_call", {})
        len_res = state["local_grep_retries"][1]
        warnings = pending.get("arguments", {}).get("warnings", [])
    
        feedback = "\n".join(f"Warning: {w}" for w in warnings)
        if feedback:
            feedback += "\n"
    
        if len_res > get_config().get("main_pipeline.grep_tool.max_results_to_refine"):
            feedback += (
                f"The grep search yielded {len_res} results, which exceeds the "
                f"limit of {get_config().get('main_pipeline.grep_tool.max_results_to_refine')}."
            )
            instructions = (
                "Please narrow the search pattern or switch to a different tool. "
                "Respond by calling one of the available tools."
            )
        else:
            feedback += "The grep search returned 0 results."
            instructions = (
                "Please broaden the search pattern or switch to a different tool. "
                "Respond by calling one of the available tools."
            )
    
        # --- Step 3: Store the follow-up prompt in state for verbose display ---
        state["follow_up_prompt"] = (
            f"Result:\n{feedback}\n\n"
            f"Instruction:\n{instructions}"
        )
    
        # --- Step 4: Continue the open planner conversation with the feedback ---
        try:
            return self.planner_llm.submit_tool_result_and_continue(
                tool_call_id=pending.get("tool_id", "unknown"),
                tool_name="grep_tool",
                result=feedback,
                next_instruction=instructions,
                tools=planner_tools,
                tool_choice="any",
            )
        except Exception as exc:
            # Model fell back to plain text — redirect to self-correction
            self.console.print(
                f"[dim][yellow]Planner fallback (grep refine): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="simple_regeneration_tool",
                tool_id="fallback",
                arguments={"advice": str(exc)},
            )
    
    
    def _plan_tree_followup(self, state, planner_tools) -> ToolCallResult:
        """
        Handle the tree-tool follow-up branch.
    
        Called when the previous tool was tree_tool.  The tree output was stored
        in state['rewritten_prompt'] by use_tree_tool; we submit it as the tool
        result so the model can decide what to search next based on the structure
        it just saw.
    
        Side-effect: removes the last history entry (same reasoning as grep
        refinement — the tree call itself is not an evidence-gathering step and
        does not need to persist as a standalone entry once the follow-up is made).
        """
        # --- Step 1: Retrieve tree output and clean up history ---
        tree_str = state["rewritten_prompt"]
        state["pipeline_state"]["execution_history"] = (
            state["pipeline_state"]["execution_history"][:-1]
        )
    
        # --- Step 2: Construct and store the follow-up prompt ---
        tree_result = (
            "Here is the directory tree (rooted at the longest valid prefix "
            f"of the path you provided):\n\n{tree_str}"
        )
        tree_instruction = "Based on the tree above, select the next tool to call."
        state["follow_up_prompt"] = (
            f"Result:\n{tree_result}\n\n"
            f"Instruction:\n{tree_instruction}"
        )
    
        # --- Step 3: Submit tree result and ask for the next tool ---
        pending = state.get("pending_tool_call", {})
        try:
            return self.planner_llm.submit_tool_result_and_continue(
                tool_call_id=pending.get("tool_id", "unknown"),
                tool_name="tree_tool",
                result=tree_result,
                next_instruction=tree_instruction,
                tools=planner_tools,
                tool_choice="any",
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (tree follow-up): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="simple_regeneration_tool",
                tool_id="fallback",
                arguments={"advice": str(exc)},
            )
        
    def _plan_rag_followup(self, state, planner_tools) -> ToolCallResult:
        """Handle the RAG-tool follow-up branch.
        
        Called when the previous tool was rag_tool but no results were found. 
        The model can decide how to proceed — whether to try grep_tool with specific 
        keywords/sources, or to reformulate the query and try RAG again."""
        # --- Step 1: Retrieve RAG output and clean up history ---
        parameters = state.get("pending_tool_call", {}).get("arguments", {})
        state["pipeline_state"]["execution_history"] = (
            state["pipeline_state"]["execution_history"][:-1]
        )
        
        # --- Step 2: Construct and store the follow-up prompt ---
        rag_result = (
            "The RAG search did not return any relevant results. "
            "Here are the parameters that were used for the search:\n"
            f"{parameters}\n\n"
        )
        rag_instruction = (
            "Based on this information, select the next tool to call. "
            "You might want to try grep_tool with specific keywords/sources, "
            "or reformulate the query and try RAG again."
        )
        state["follow_up_prompt"] = (
            f"Result:\n{rag_result}\n"
            f"Instruction:\n{rag_instruction}"
        )
        
        # --- Step 3: Submit RAG result and ask for the next tool ---
        pending = state.get("pending_tool_call", {})
        try:
            return self.planner_llm.submit_tool_result_and_continue(
                tool_call_id=pending.get("tool_id", "unknown"),
                tool_name="rag_tool",
                result=rag_result,
                next_instruction=rag_instruction,
                tools=planner_tools,
                tool_choice="any",
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (RAG follow-up): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="simple_regeneration_tool",
                tool_id="fallback",
                arguments={"advice": str(exc)},
            )
    
    def _plan_prior_evidence_followup(self, state, planner_tools) -> ToolCallResult:
        """Handle the prior evidence-tool follow-up branch.
        Called when the previous tool was prior_evidence_tool but no results were found."""
        # --- Step 1: Retrieve prior evidence output and clean up history ---
        parameters = state.get("pending_tool_call", {}).get("arguments", {})
        state["pipeline_state"]["execution_history"] = (
            state["pipeline_state"]["execution_history"][:-1]
        )
        
        # --- Step 2: Construct and store the follow-up prompt ---
        prior_evidence_result = (
            "The prior evidence search did not return any relevant results. "
            "Here are the parameters that were used for the search:\n"
            f"{parameters}\n\n"
        )
        prior_evidence_instruction = (
            "Based on this information, select the next tool to call."
        )
        state["follow_up_prompt"] = (
            f"Result:\n{prior_evidence_result}\n"
            f"Instruction:\n{prior_evidence_instruction}"
        )
        
        # --- Step 3: Submit prior evidence result and ask for the next tool ---
        pending = state.get("pending_tool_call", {})
        try:
            return self.planner_llm.submit_tool_result_and_continue(
                tool_call_id=pending.get("tool_id", "unknown"),
                tool_name="prior_evidence_tool",
                result=prior_evidence_result,
                next_instruction=prior_evidence_instruction,
                tools=planner_tools,
                tool_choice="any",
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (prior evidence follow-up): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="simple_regeneration_tool",
                tool_id="fallback",
                arguments={"advice": str(exc)},
            )
    
    def _plan_graph_followup(self, state, planner_tools) -> ToolCallResult:
        """
        Handle the graph-tool follow-up branch.
        
        Called when the previous tool was graph_tool with an action other than 'search'.
        The graph output was stored in state['rewritten_prompt'] by use_graph_tool;
        we submit it as the tool result so the model can decide what graph navigation
        to do next based on the structure it just explored.
        
        Side-effect: removes the last history entry (graph navigation itself is not
        evidence-gathering, just structural exploration).
        """
        # --- Step 1: Retrieve graph output and clean up history ---
        graph_str = state["rewritten_prompt"]
        state["pipeline_state"]["execution_history"] = (
            state["pipeline_state"]["execution_history"][:-1]
        )
        
        # --- Step 2: Construct and store the follow-up prompt ---
        graph_result = (
            "Here are the structural graph results from your navigation:\n\n"
            f"{graph_str}"
        )
        warning_str = ""
        if state.get("local_graph_retries", 0) >= get_config().get("main_pipeline.graph_tool.max_retries", 5)-3:
            warning_str = f". But beware, you only have {state.get('local_graph_retries', 0)} attempts left for graph navigation !"
        graph_instruction = (
            "Based on these structural results, decide your next action:\n"
            f"- Continue graph navigation with another action (tree, node, neighbors, edges){warning_str}\n"
            "- Or switch to rag_tool/grep_tool to examine specific code\n"
            "When you're ready to conclude graph exploration, use the 'search' action."
        )
        state["follow_up_prompt"] = (
            f"Result:\n{graph_result}\n\n"
            f"Instruction:\n{graph_instruction}"
        )
        
        # --- Step 3: Submit graph result and ask for the next tool ---
        pending = state.get("pending_tool_call", {})
        try:
            return self.planner_llm.submit_tool_result_and_continue(
                tool_call_id=pending.get("tool_id", "unknown"),
                tool_name="graph_tool",
                result=graph_result,
                next_instruction=graph_instruction,
                tools=planner_tools,
                tool_choice="any",
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (graph follow-up): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="simple_regeneration_tool",
                tool_id="fallback",
                arguments={"advice": str(exc)},
            )

    
    
    def _plan_normal(self, state, planner_tools, is_continuation) -> Tuple[ToolCallResult, str, str]:
        """
        Handle the normal planning branch (kickoff or continuation).
    
        Returns a tuple of (tool_call, system_prompt, reasoning) so that the
        caller can use them for verbose display without re-computing.
    
        Kickoff  — single generate_with_tools call; reasoning is empty.
        Continuation — two-step chain-of-thought:
            1. generate_response  → model writes explicit reasoning (forces the
                                    flowchart to be executed step by step rather
                                    than pattern-matched in a single forward pass)
            2. generate_with_tools → model picks a tool, constrained by step 1's
                                    reasoning already in its context window
        """
        reasoning = ""
    
        if not is_continuation:
            # --- Kickoff: no prior context, single tool-calling call ---
            system_prompt = self._kickoff_system_prompt(state)
            self.planner_llm.reset_context()
            try:
                tool_call = self.planner_llm.generate_with_tools(
                    user_message="Begin the investigation. Select the best first tool to use.",
                    tools=planner_tools,
                    system_prompt=system_prompt,
                    tool_choice="any",
                )
            except Exception as exc:
                self.console.print(
                    f"[dim][yellow]Planner fallback (kickoff): {exc}[/yellow][/dim]"
                )
                tool_call = ToolCallResult(
                    tool_name="simple_regeneration_tool",
                    tool_id="fallback",
                    arguments={"advice": (
                        "The planner failed to select a valid tool. "
                        "Please re-examine the question and select the most appropriate tool."
                    )},
                )
            return tool_call, system_prompt, reasoning
    
        # --- Continuation: two-step chain-of-thought ---
        system_prompt = self._continuation_system_prompt(state)
        self.planner_llm.reset_context()
    
        # Step 1 — Free-text reasoning.
        # Forcing the model to write out each step of the flowchart explicitly
        # prevents it from pattern-matching directly to submit_answer based on
        # the surface form of the Proposed Solution.  The written reasoning
        # becomes prior context that constrains the tool selection in step 2.
        try:
            reasoning = self.planner_llm.generate_response(
                user_message=(
                    "Follow the flowchart in your instructions step by step. "
                    "Write out your reasoning for each step explicitly, "
                    "then state which tool you will call and why."
                ),
                system_prompt=system_prompt,
                temperature=0.2,
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (continuation reasoning): {exc}[/yellow][/dim]"
            )
            return ToolCallResult(
                tool_name="submit_answer",
                tool_id="fallback",
                arguments={
                    "thought": (
                        "Planner reasoning step failed due to provider error; "
                        "proceeding with current answer."
                    )
                },
            ), system_prompt, reasoning
    
        # Step 2 — Forced tool call.
        # The model's own reasoning from step 1 is now in self.context; it cannot
        # easily contradict it without being incoherent within the same window.
        try:
            tool_call = self.planner_llm.generate_with_tools(
                user_message="Based on your reasoning above, call the appropriate tool now.",
                tools=planner_tools,
                tool_choice="any",
                # system_prompt intentionally omitted — already in context from step 1
            )
        except Exception as exc:
            self.console.print(
                f"[dim][yellow]Planner fallback (continuation): {exc}[/yellow][/dim]"
            )
            tool_call = ToolCallResult(
                tool_name="submit_answer",
                tool_id="fallback",
                arguments={
                    "thought": (
                        "Planner tool-selection step failed due to provider error; "
                        "proceeding with current answer."
                    )
                },
            )
    
        return tool_call, system_prompt, reasoning

    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        Planner node — decides which tool to call next.
    
        Dispatches to one of three private sub-methods depending on context:
        _plan_grep_refinement  — previous grep had 0 or too many results
        _plan_tree_followup    — previous call was tree_tool
        _plan_normal           — all other cases (kickoff or continuation)
    
        After the tool is chosen, the method:
        1. Validates the tool name and redirects invalid choices to
            simple_regeneration_tool.
        2. Updates state with the pending tool call and current thought.
        3. Renders verbose output showing the prompt, reasoning, and decision.
        4. Runs lazy distillation on any raw results left by the previous tool.
        """
        self.console.print("[dim]--- SUB-NODE: Agentic Router (Planner) ---[/dim]")
    
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        # If in interactive mode we must distill the last results
        if state["pipeline_state"].get("undistilled_log"):
            self.console.print("[dim]Distilling last query results...[/dim]")
            new_facts = self._distill_results(state, state["pipeline_state"].get("undistilled_log"))
            state["pipeline_state"]["undistilled_log"] = None  # Clear after distillation
            if state["pipeline_state"]["verbose"]:
                self.console.print(f"[dim]Extracted {len(new_facts)} facts.[/dim]")
    
        # ------------------------------------------------------------------
        # Step 1: Determine context and build the tool schema list
        # ------------------------------------------------------------------
        history = self._get_history(state)
        is_continuation = state.get("continuation")
        if is_continuation is None:
            is_continuation = len(history) > 0
        state["continuation"] = is_continuation

        # submit_answer is only offered after at least one search step so the
        # model cannot exit on the very first turn without gathering any evidence.
        include_grade = is_continuation
        planner_tools = _build_planner_tools(
            tools=[self.rag_tool, self.grep_tool, self.graph_tool,
                   self.tree_tool, self.script_finder_tool, self.prior_evidence_tool],
            include_grade=include_grade,
        )
    
        # ------------------------------------------------------------------
        # Step 2: Dispatch to the appropriate planning sub-method
        # ------------------------------------------------------------------
        already_distilled = False  # True for grep/tree branches which handle their own cleanup
    
        grep_refinement_active = (
            "local_grep_retries" in state
            and state["pipeline_state"].get("execution_history")
            and state["pipeline_state"]["execution_history"][-1]["tool"] == "grep_tool"
        )
        tree_followup_active = (
            history
            and history[-1]["tool"] == "tree_tool"
            and state.get("rewritten_prompt")
        )

        rag_followup_active = (
            history
            and history[-1]["tool"] == "rag_tool"
            and state.get("rewritten_prompt")
        )
        prior_evidence_followup_active = (
            history
            and history[-1]["tool"] == "prior_evidence_tool"
            and state.get("rewritten_prompt")
        )
        
        # Graph tool follow-up: continue loop if action was NOT "search"
        graph_followup_active = (
            history
            and history[-1]["tool"] == "graph_tool"
            and state.get("rewritten_prompt")
            and state.get("pending_tool_call", {}).get("arguments", {}).get("action") != "search"
        )
        
        system_prompt = ""
        reasoning = ""
    
        if grep_refinement_active:
            already_distilled = True  # grep refinement never has raw results to distil
            tool_call = self._plan_grep_refinement(state, planner_tools)
    
        elif tree_followup_active:
            already_distilled = True  # tree output is not evidence, nothing to distil
            tool_call = self._plan_tree_followup(state, planner_tools)
        
        elif graph_followup_active:
            already_distilled = True  # graph navigation is structural, not evidence to distil
            tool_call = self._plan_graph_followup(state, planner_tools)
        
        elif rag_followup_active:
            already_distilled = True  # RAG follow-up is triggered by a negative result, so no distillation needed
            tool_call = self._plan_rag_followup(state, planner_tools)

        elif prior_evidence_followup_active:
            already_distilled = True  # prior evidence follow-up is triggered by a negative result, so no distillation needed
            tool_call = self._plan_prior_evidence_followup(state, planner_tools)
    
        else:
            tool_call, system_prompt, reasoning = self._plan_normal(
                state, planner_tools, is_continuation
            )
    
        # ------------------------------------------------------------------
        # Step 3: Validate the chosen tool
        # ------------------------------------------------------------------
        chosen_tool = tool_call.tool_name
        arguments = tool_call.arguments
        tool_id = tool_call.tool_id
    
        if chosen_tool not in VALID_TOOLS:
            self.console.print(
                f"  [System Warning]: Unknown tool '{chosen_tool}' → redirecting to regeneration."
            )
            chosen_tool = "simple_regeneration_tool"
            arguments = {
                "advice": (
                    f"SYSTEM ERROR: The tool '{tool_call.tool_name}' does not exist. "
                    f"Valid tools: {', '.join(VALID_TOOLS)}."
                )
            }
    
        # Extract the thought field the model may have included in arguments,
        # falling back to a generic label for the history log.
        thought = arguments.pop("thought", f"Calling {chosen_tool}.")
        param_summary = json.dumps(arguments, ensure_ascii=False)
    
        # ------------------------------------------------------------------
        # Step 4: Update state with the tool decision
        # ------------------------------------------------------------------
        state['current_thought'] = thought
        state['pending_tool_call'] = {
            "tool_id": tool_id,
            "tool_name": chosen_tool,
            "arguments": arguments,
        }
    
        max_retries = self.config_manager.get("main_pipeline.agent_logic.max_retries", 5)
        state["regenerate"] = (
            chosen_tool != "submit_answer"
            and state["pipeline_state"]["retry_count"] <= max_retries
        )
    
        # ------------------------------------------------------------------
        # Step 5: Verbose display
        # ------------------------------------------------------------------
        if state["pipeline_state"]["verbose"]:
            # Check if there's a follow-up prompt to display
            if state.get("follow_up_prompt"):
                prompt_content = Panel(
                    escape(state["follow_up_prompt"]), title="Follow-up Prompt", border_style="bright_green"
                )
            else:
                prompt_content = Panel(
                    escape(system_prompt), title="Planner Prompt", border_style="purple"
                )
    
            panels = [prompt_content]
    
            if reasoning:
                panels.append(Panel(
                    Markdown(reasoning), title="Planner Reasoning", border_style="yellow"
                ))
    
            panels.append(Text.from_markup(
                f"\nPlanner selected tool: [bold green]{chosen_tool}[/bold green] "
                f"with arguments: [bold orange3]{escape(param_summary)}[/bold orange3]\n"
            ))
            panels.append(Panel(
                Markdown(thought), title="Planner Thought", border_style="blue"
            ))
    
            self.console.print(Panel(
                Group(*panels),
                title=f"Planner - {self.planner_llm.model_name}",
                border_style="bright_red",
            ))
    
        # ------------------------------------------------------------------
        # Step 6: Lazy distillation of previous tool results
        # ------------------------------------------------------------------
        # Raw results are stored in execution_history by the tool nodes and
        # distilled here (lazily, at the start of the next planning turn) so
        # that the distillation LLM can use the Main Agent's generation as
        # additional context when deciding what to keep.
        exec_history = state["pipeline_state"].get("execution_history", [])
        if exec_history and not already_distilled:
            prev_results = exec_history[-1].get("results_to_analyse")
            if prev_results and chosen_tool != "submit_answer":
                self.console.print("[dim]Distilling previous tool results...[/dim]")
                
                new_facts = self._distill_results(state, exec_history[-1])
                
                exec_history[-1]["outcome_summary"] += (
                    f" Extracted {len(new_facts)} relevant facts."
                )

                if state["pipeline_state"]["verbose"]:
                    self.console.print(f"[dim]Extracted {len(new_facts)} facts.[/dim]")
    
        # Clear follow_up_prompt after verbose display to avoid stale data in future iterations
        state.pop("follow_up_prompt", None)
    
        return state

    # -----------------------------------------------------------------------
    # Tool nodes  (read clean args from pending_tool_call)
    # -----------------------------------------------------------------------
    
    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:
        """Semantic search — parameters arrive as clean JSON from the tool call."""
        self.console.print("[dim]--- SUB-NODE: RAG Tool ---[/dim]")
    
        # ------------------------------------------------------------------
        # Step 1: Unpack arguments from the pending tool call
        # ------------------------------------------------------------------
        args = state.get("pending_tool_call", {}).get("arguments", {})
        query: str = args.get("query", "")
        key_words: Optional[List[str]] = args.get("key_words") or None
        sources = args.get("sources") or None
        thought: str = state.get("current_thought", "No reasoning provided.")
    
        if state['pipeline_state']['verbose']:
            if key_words:
                self.console.print(
                    f"[dim]Key words: [bright_magenta]{escape(str(key_words))}[/bright_magenta][/dim]"
                )
            if sources:
                self.console.print(
                    f"[dim]Sources: [sea_green1]{escape(str(sources))}[/sea_green1][/dim]"
                )
    
        # ------------------------------------------------------------------
        # Step 2: Execute the retrieval
        # ------------------------------------------------------------------
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
    
        # ------------------------------------------------------------------
        # Step 3: Record in execution history
        # Raw results are stored in results_to_analyse so that the next
        # planning turn can distil them lazily (with the Main Agent's generation
        # available as additional context at that point).
        # ------------------------------------------------------------------
        outcome_str = f"Retrieved {len(results)} chunks."
        self._append_history(state, "rag_tool", args, outcome_str, thought, results)
    
        # ------------------------------------------------------------------
        # Step 4: Format raw results for the Solver prompt
        # Group results by source file with line numbers
        # ------------------------------------------------------------------
        raw_results_str = self._format_results(results)
    
        # ------------------------------------------------------------------
        # Step 5: Assemble the rewritten prompt for the Solver
        # ------------------------------------------------------------------
        base_prompt = self._design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### RAG RESULTS\n"
            f"The RAG tool retrieved {len(results)} relevant chunks for the query '{query}':\n"
            f"{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            "Using the RAG results above and all previous knowledge, answer the question "
            "as concisely as possible."
        )
        return state
    
    def use_grep_tool(self, state: WorkflowState) -> WorkflowState:
        """Text grep — parameters arrive as clean JSON from the tool call."""
        self.console.print("[dim]--- SUB-NODE: Grep Tool ---[/dim]")
    
        # ------------------------------------------------------------------
        # Step 1: Unpack and validate arguments from the pending tool call
        # ------------------------------------------------------------------
        args: Dict = state.get("pending_tool_call", {}).get("arguments", {})
        args.setdefault("warnings", [])  # accumulate non-fatal warnings for the planner
        pattern: str = args.get("pattern", "")
        source_regex: Optional[str] = args.get("sources") or None
        thought: str = state.get("current_thought", "No reasoning provided.")
    
        # Convert comma-separated block_type string to BlockType enum list.
        # Invalid values are recorded as warnings rather than raising, so the
        # search still runs on valid types or falls back to searching all types.
        block_type_raw: Optional[List[str]] = args.get("block_type") or None
        block_types: Optional[List["BlockType"]] = None
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
                args["warnings"].append(
                    "All given block types are invalid — block type filter was ignored."
                )
                block_types = None
            elif invalid_bt:
                args["warnings"].append(
                    f"The following block types are invalid: {', '.join(invalid_bt)}"
                )
    
        # ------------------------------------------------------------------
        # Step 2: Execute the grep search
        # ------------------------------------------------------------------
        results = self.grep_tool.search(
            pattern=pattern,
            source_regex=source_regex,
            bloc_type=block_types,
            warnings=args["warnings"],
        )
    
        # Track retry count for the refine_grep conditional edge.
        # First call initialises the counter; subsequent calls increment it.
        if "local_grep_retries" not in state:
            state["local_grep_retries"] = (0, len(results))
        else:
            state["local_grep_retries"] = (state["local_grep_retries"][0] + 1, len(results))
    
        if state['pipeline_state']['verbose']:
            self.console.print(
                f"Grep found {len(results)} matches for pattern: '{escape(pattern)}'"
            )
    
        # ------------------------------------------------------------------
        # Step 3: Truncate results if above the display limit
        # ------------------------------------------------------------------
        original_result_count = len(results)
        max_res = get_config().get("main_pipeline.grep_tool.max_results")
        res_truncated = len(results) > max_res
        if res_truncated:
            results = results[:max_res]
            if state['pipeline_state']['verbose']:
                self.console.print(f"Only the first {max_res} will be analysed")
    
        # If still 0 results after max retries, record the negative result
        # explicitly in the knowledge bank so the Solver is aware of it.
        max_grep_retries = self.config_manager.get("main_pipeline.grep_tool.max_grep_retries", 3)
        if not results and state["local_grep_retries"][0] >= max_grep_retries:
            state['pipeline_state'].setdefault("knowledge_bank", []).append(
                KnowledgeElement(
                    fact=f"No matches found for pattern '{pattern}'"
                    + (f" in sources matching '{source_regex}'." if source_regex else "."),
                    tool="grep_tool",
                    query=state['pipeline_state']["question"],
                    evidence_ids=[]
                )
            )
    
        # ------------------------------------------------------------------
        # Step 4: Record in execution history
        # Raw results are stored for lazy distillation on the next planning turn.
        # ------------------------------------------------------------------
        total_sources = {r.chunk.metadata.get('original_file_path', 'Unknown') for r in results}
        outcome_str = (
            f"Grep found {original_result_count} matches from {len(total_sources)} distinct scripts."
        )
        if res_truncated:
            outcome_str += f" Only the first {max_res} were analyzed."
        self._append_history(state, "grep_tool", args, outcome_str, thought, results)
    
        # ------------------------------------------------------------------
        # Step 5: Compact results to fit within the context line budget
        # ------------------------------------------------------------------
        max_lines = self.config_manager.get("main_pipeline.grep_tool.max_lines")
        shortened_res = self.grep_tool.shorten_results(
            pattern, results, max_lines
        )
        if len(shortened_res) < len(results):
            raise Exception("Lost results during compaction")
    
        # Group results by source file with line numbers for the prompt
        raw_results_str = self._format_results(shortened_res)
    
        # ------------------------------------------------------------------
        # Step 6: Assemble the rewritten prompt for the Solver
        # ------------------------------------------------------------------
        base_prompt = self._design_first_part_prompt(state)
        state['rewritten_prompt'] = f"{base_prompt}### GREP RESULTS\n"
    
        if res_truncated:
            state['rewritten_prompt'] += (
                f"WARNING: Results truncated — only the first {max_res} "
                f"out of {original_result_count} are shown.\n"
            )
        elif not results:
            state['rewritten_prompt'] += (
                f"WARNING: No matches for pattern '{pattern}'"
                + (f" in sources matching '{source_regex}'.\n" if source_regex else ".\n")
            )
        else:
            # Inform the Solver of any active filters so it can cite them
            ignored_source_warning = "No files in the codebase match the source_regex so source filter was ignored."
            if source_regex and ignored_source_warning not in args["warnings"]:
                state['rewritten_prompt'] += (
                    f"Search restricted to files whose path matches: {source_regex}\n"
                )
            state['rewritten_prompt'] += (
                f"The Grep tool found {len(results)} matches for '{pattern}' "
                f"from {len(total_sources)} distinct scripts."
            )
    
        state['rewritten_prompt'] += (
            f"\n\n{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            f"Using the {len(results)} grep results (from {len(total_sources)} distinct scripts) above and all previous knowledge, answer the "
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

        knowledge_elements = [
            KnowledgeElement(
                fact=summary,
                tool="script_finder_tool",
                query=state["pipeline_state"]["question"],
                evidence_ids=[]
            ) for summary in new_facts
        ]
        state['pipeline_state'].setdefault("knowledge_bank", []).extend(knowledge_elements)

        outcome_str = f"Found {len(found_paths)} scripts. Content read and distilled."
        if len(found_paths) < len(script_names):
            outcome_str += (
                f" {len(script_names) - len(found_paths)} could not be found."
                f" Only found: {[self.mapping.get(os.path.splitext(os.path.basename(p))[0], p) for p in found_paths]}."
            )
        self._append_history(state, "script_finder_tool", script_names, outcome_str, thought)

        base_prompt = self._design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
        )
        if found_paths:
            state['rewritten_prompt'] += (
                f"The scripts {[self.mapping.get(os.path.splitext(os.path.basename(p))[0], p) for p in found_paths]} have been located and analyzed, "
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

    def use_prior_evidence_tool(self, state: WorkflowState) -> WorkflowState:
        """Retrieve prior evidence from accumulated evidence collected in previous tool calls."""
        self.console.print("[dim]--- SUB-NODE: Prior Evidence Tool ---[/dim]")
        
        # Step 1: Unpack arguments
        args = state.get("pending_tool_call", {}).get("arguments", {})
        evidence_ids = args.get("evidence_ids", [])
        thought = state.get("current_thought", "No reasoning provided.")
        
        if state['pipeline_state']['verbose']:
            self.console.print(
                f"[dim]Evidence IDs requested: [bright_magenta]{escape(str(evidence_ids))}[/bright_magenta][/dim]"
            )
        
        # Step 2: Retrieve prior evidence using the tool's method
        accumulated_evidence = state['pipeline_state'].get("accumulated_evidence", {})
        results_by_id = self.prior_evidence_tool.retrieve_prior_evidence(
            evidence_ids=evidence_ids,
            accumulated_evidence=accumulated_evidence
        )
        
        # Count total results and track which IDs failed
        total_results = sum(len(batch) for batch in results_by_id.values())
        missing_ids = [eid for eid, batch in results_by_id.items() if not batch]
        
        if state['pipeline_state']['verbose']:
            self.console.print(f"[dim]Retrieved {total_results} prior evidence items[/dim]")
            if missing_ids:
                self.console.print(f"[dim]Missing IDs: {missing_ids}[/dim]")
        
        # Step 3: Record in execution history
        outcome_str = f"Retrieved {total_results} prior evidence items from {len(evidence_ids)} requested IDs"
        if missing_ids:
            outcome_str += f" (IDs not found: {missing_ids})"
        self._append_history(state, "prior_evidence_tool", evidence_ids, outcome_str, thought)
        
        # Step 4: Format results using the tool's method
        _, raw_results_str = self.prior_evidence_tool.format_results_by_source(results_by_id)
        
        # Step 5: Assemble the rewritten prompt for the Solver
        base_prompt = self._design_first_part_prompt(state)
        state['rewritten_prompt'] = f"{base_prompt}### PRIOR EVIDENCE RESULTS\n"
        
        if total_results == 0:
            state['rewritten_prompt'] += (
                f"WARNING: No evidence found for requested IDs: {evidence_ids}\n"
            )
        else:
            state['rewritten_prompt'] += (
                f"Retrieved {total_results} chunks from prior investigations "
                f"(Evidence IDs: {', '.join(evidence_ids)}).\n"
            )
        
        state['rewritten_prompt'] += (
            f"\n\n{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            "Using the prior evidence results above and all previous knowledge, answer the "
            "question as concisely as possible."
        )
        return state

    def use_graph_tool(self, state: WorkflowState) -> WorkflowState:
        """Use the structural graph tool for dependency navigation."""
        self.console.print("[dim]--- SUB-NODE: Graph Tool ---[/dim]")
        args = state.get("pending_tool_call", {}).get("arguments", {})
        action = args.get("action")
        graph_retries = state.get("local_graph_retries", 0)
        thought = state.get("current_thought", "No reasoning provided.")

        if not action:
            raise ValueError("graph_tool requires an 'action' argument")

        result = self.graph_tool.execute(**args)

        stats = result.get("stats", {}) if isinstance(result, dict) else {}
        if isinstance(stats, dict) and stats.get("error"):
            outcome_str = f"Graph action '{action}' failed: {result.get('error', 'Unknown error')}"
        else:
            outcome_str = f"Graph action '{action}' executed successfully."

        self._append_history(state, "graph_tool", args, outcome_str, thought)

        graph_result_text = self.graph_tool.to_prompt_text(result)
        state["local_graph_retries"] = graph_retries + 1
        if action == "search":
            base_prompt = self._design_first_part_prompt(state)
            state["rewritten_prompt"] = (
                f"{base_prompt}"
                "### GRAPH RESULTS\n"
                f"Action: {action}\n"
                f"Parameters: {json.dumps(args, ensure_ascii=False)}\n\n"
                f"{graph_result_text}\n\n"
                "### INSTRUCTION\n"
                "Use these structural graph results together with verified facts to answer "
                "the question as concisely as possible."
            )
        else:
            state["rewritten_prompt"] = graph_result_text  # For non-search actions, just update the prompt with the new graph state for the next turn's planning
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
    
    def refine_rag(self, state: WorkflowState) -> str:
        self.console.print("[dim]--- SUB-DECISION: RAG results validation ---[/dim]")
        # For simplicity, we only check if any results were returned; more complex
        # validation could be implemented as needed.
        exec_history = state["pipeline_state"].get("execution_history", [])
        if not exec_history or exec_history[-1]["tool"] != "rag_tool":
            self.console.print("[dim]    -> No recent RAG call found. Validated by default.[/dim]")
            return "validated"

        last_results = exec_history[-1].get("results_to_analyse", [])
        if last_results:
            self.console.print(f"[dim]    -> RAG returned {len(last_results)} results, validated.[/dim]")
            return "validated"
        else:
            self.console.print("[dim]    -> RAG returned 0 results, replanning.[/dim]")
            return "replan"
    
    def refine_prior_evidence(self, state: WorkflowState) -> str:
        """Decide whether prior evidence results are satisfactory or need replanning."""
        self.console.print("[dim]--- SUB-DECISION: Prior evidence results validation ---[/dim]")
        exec_history = state["pipeline_state"].get("execution_history", [])
        if not exec_history or exec_history[-1]["tool"] != "prior_evidence_tool":
            self.console.print("[dim]    -> No recent prior evidence call found. Validated by default.[/dim]")
            return "validated"

        last_results = exec_history[-1].get("results_to_analyse", [])
        if last_results:
            self.console.print(f"[dim]    -> Prior evidence returned {len(last_results)} results, validated.[/dim]")
            return "validated"
        else:
            self.console.print("[dim]    -> Prior evidence returned 0 results, replanning.[/dim]")
            return "replan"

    def should_continue_graph_navigation(self, state: WorkflowState) -> str:
        """
        Decide whether to continue graph navigation or end the loop.
        
        Returns "continue" to loop back for more navigation,
        "end" to terminate if search action was used.
        """
        self.console.print("[dim]--- SUB-DECISION: Graph navigation continuation ---[/dim]")
        # Check the pending tool call to see what action was executed
        pending = state.get("pending_tool_call", {})
        action = pending.get("arguments", {}).get("action")
        
        # If action was "search", end the loop (user found what they need)
        if action == "search":
            self.console.print("[dim]    -> 'search' action used, concluding graph exploration.[/dim]")
            return "end"
        
        if  state.get("local_graph_retries", 0) >= self.config_manager.get("main_pipeline.graph_tool.max_graph_iterations", 5):
            self.console.print("[dim]    -> Max graph iterations reached, concluding graph exploration.[/dim]")
            return "end"
        
        # Otherwise, continue for more graph navigation
        self.console.print(f"[dim]    -> Action '{action}' allows continued navigation.[/dim]")
        return "continue"

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------

    def build_graph(self):
        self.add_node("agentic_router", self.agentic_router)
        self.add_node("rag_tool", self.use_rag_tool)
        self.add_node("grep_tool", self.use_grep_tool)
        self.add_node("graph_tool", self.use_graph_tool)
        self.add_node("tree_tool", self.use_tree_tool)
        self.add_node("script_finder_tool", self.use_script_finder_tool)
        self.add_node("prior_evidence_tool", self.use_prior_evidence_tool)
        self.add_node("simple_regeneration_tool", self.use_simple_regeneration_tool)

        self.add_edge(START, "agentic_router")
        self.add_conditional_edges(
            "agentic_router",
            self.decide_after_routing,
            {
                "rag_tool": "rag_tool",
                "grep_tool": "grep_tool",
                "graph_tool": "graph_tool",
                "tree_tool": "tree_tool",
                "script_finder_tool": "script_finder_tool",
                "prior_evidence_tool": "prior_evidence_tool",
                "simple_regeneration_tool": "simple_regeneration_tool",
                "submit_answer": END,
            },
        )
        self.add_conditional_edges(
            "grep_tool",
            self.refine_grep,
            {"replan": "agentic_router", "validated": END},
        )
        self.add_edge("tree_tool", "agentic_router")
        self.add_conditional_edges(
            "rag_tool",
            self.refine_rag,
            {"replan": "agentic_router", "validated": END}
        )
        self.add_conditional_edges(
            "prior_evidence_tool",
            self.refine_prior_evidence,
            {"replan": "agentic_router", "validated": END}
        )
        self.add_conditional_edges(
            "graph_tool",
            self.should_continue_graph_navigation,
            {"continue": "agentic_router", "end": END},
        )
        self.add_edge("script_finder_tool", END)
        self.add_edge("simple_regeneration_tool", END)

        return self.compile()
        
