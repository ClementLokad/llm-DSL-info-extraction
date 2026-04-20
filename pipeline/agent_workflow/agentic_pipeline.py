"""
AgenticPipeline
===============
LangGraph pipeline that orchestrates the agentic workflow sub-graph,
a main LLM Solver, and a Cleaning LLM.

Role conventions (matching Mistral training corpus)
----------------------------------------------------
  system      – static identity / persona injected once at the start of
                every LLM call.
  user        – the human-visible prompt (question + context + facts).
  assistant   – the model's reply.
  tool        – the result of a function call (handled in MistralAgent).

The *Solver* (generate_answer) and *Cleaner* (clean_generated_answer) now
use generate_response(user_message, system_prompt=...) so that the static
persona lands in the "system" role rather than being concatenated into the
user turn.  This matches how instruction-tuned Mistral models were trained
and improves instruction-following and output consistency.
"""

from __future__ import annotations

import re
import time

from langgraph.graph import END, StateGraph, START
from config_manager import get_config
from pipeline.agent_workflow.workflow_base import BaseAgentWorkflow
from pipeline.langgraph_base import AgentGraphState, BasePipeline, GraphState
from pipeline.answer_validation import (
    SourcePathValidator,
    build_validation_feedback,
    append_validation_warning,
)
from pipeline.stats_collector import get_collector

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.markup import escape

# Reuse prepare_agent from the existing helper
from agents.prepare_agent import prepare_agent


# ---------------------------------------------------------------------------
# Static system prompts
# ---------------------------------------------------------------------------


with open("base_instructions.txt", "r") as file:
    base_instructions = file.read()

_SOLVER_SYSTEM_PROMPT = base_instructions + (
    "\n### SYSTEM ROLE\n"
    "You are an expert technical assistant specialised in the **Lokad Envision** codebase.\n"
    "Lokad is a supply chain optimisation company. Envision is their domain-specific "
    "programming language for quantitative supply chain logic and probabilistic forecasting.\n"
    "Answer the user's question fully and accurately, citing the relevant source files and the corresponding lines.\n"
    "If you do not have sufficient information to answer the question, DO NOT try to answer it."
    "Specify rather what is missing.\n"
    "Do NOT add conversational filler or repeat the question.\n\n"
)

_CLEANER_SYSTEM_PROMPT = (
    "You are a post-processing assistant. Your only job is to clean and reformat a raw "
    "LLM-generated answer into a concise, well-structured final answer.\n"
    "Rules:\n"
    "  - Remove internal thoughts, tool-usage notes, or planning artefacts.\n"
    "  - Translate the answer if needed so it is in the SAME language as the question.\n"
    "  - Write numbers with digits not letters (e.g. '5' not 'five').\n"
    "  - Cite sources (full file paths) where relevant.\n"
    "  - Do NOT add conversational filler.\n"
    "  - Respond strictly in this XML format:\n"
    "    <final_answer>[Your cleaned answer here]</final_answer>"
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AgenticPipeline(BasePipeline):
    """
    LangGraph pipeline wrapping the agentic workflow sub-graph.

    generate_answer  → calls the Solver with a proper system prompt.
    clean_generated_answer → calls the Cleaner with a proper system prompt.
    """

    def __init__(self, console: Console, agent: BaseAgentWorkflow):
        super().__init__(console)
        validation_cfg = get_config().get("main_pipeline.answer_validation", {})
        self.main_llm = prepare_agent(
            get_config().get(
                'main_pipeline.agent_logic.main_llm',
                get_config().get_default_agent(),
            )
        )
        self.cleaning_llm = prepare_agent(
            get_config().get(
                'main_pipeline.agent_logic.cleaning_llm',
                get_config().get_default_agent(),
            )
        )
        self.rate_limit_delay = get_config().get('agent.rate_limit_delay', 0)
        self.agent = agent.build_graph()
        self.benchmark_type = get_config().get_benchmark_type()
        self.answer_validation_cfg = validation_cfg
        self.path_validator = SourcePathValidator(
            ignore_extension=validation_cfg.get("ignore_extension", True),
            ignore_leading_slash=validation_cfg.get("ignore_leading_slash", True),
            allow_partial_suffix_match=validation_cfg.get("allow_partial_suffix_match", True),
            ignore_data_extensions=validation_cfg.get("ignore_data_extensions", True),
            ignored_path_extensions=validation_cfg.get("ignored_path_extensions", ["ion", "csv"]),
        )

    # -----------------------------------------------------------------------
    # Agentic workflow node
    # -----------------------------------------------------------------------

    def run_agentic_workflow(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Agentic Workflow'
        Invokes the sub-graph defined in workflow_base / concrete_workflow.
        """
        self.console.print("[dim]--- NODE: Agentic Workflow (Subgraph) ---[/dim]")

        sub_input = {
            "pipeline_state": state,
            "regenerate": False,
            "error": None,
            "rewritten_prompt": None,
            "current_thought": None,
            "pending_tool_call": None,
        }

        final_sub_state = self.agent.invoke(sub_input, {"recursion_limit": 50})
        updated_pipeline_state = final_sub_state["pipeline_state"]

        new_prompt = final_sub_state.get("rewritten_prompt")
        if not new_prompt:
            new_prompt = f"Question: {state['question']}"

        return {
            "knowledge_bank": updated_pipeline_state.get("knowledge_bank", []),
            "execution_history": updated_pipeline_state.get("execution_history", []),
            "accumulated_evidence": updated_pipeline_state.get("accumulated_evidence", {}),
            "prompt": new_prompt,
            "regenerate_needed": final_sub_state["regenerate"],
            "retry_count": state["retry_count"],
            "undistilled_log": updated_pipeline_state.get("undistilled_log", []),
        }

    # -----------------------------------------------------------------------
    # Logic checker node
    # -----------------------------------------------------------------------

    def check_agent_logic(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Logic Checker (Agentic)'
        Decides whether to loop back for more retrieval or proceed to cleaning.
        """
        self.console.print("[dim]--- NODE: Check Agent Logic ---[/dim]")

        if not state.get("generation"):
            self.console.print("[dim]    -> First pass detected. Forcing generation.[/dim]")
            return {"regenerate_needed": True}

        if state["regenerate_needed"]:
            tool_name = (state['execution_history'][-1]['tool']
                         if state.get("execution_history") else "unknown")
            self.console.print(
                f"[dim]    -> Agent used tool '{tool_name}'. Regenerating answer.[/dim]"
            )
            return {"retry_count": state["retry_count"] + 1}

        self.console.print("[dim]    -> Agent satisfied. Proceeding to cleaning.[/dim]")
        return {"final_answer": state["generation"]}

    # -----------------------------------------------------------------------
    # Routing decision
    # -----------------------------------------------------------------------

    def decide_after_logic_check(self, state: GraphState) -> str:
        """
        Decision Point: After 'Logic Checker'
        Determines the next step after the logic check.
        - 'if error detected': Returns to 'agentic_workflow' (loop).
        - 'else': Continues to 'clean_generated_answer'.
        """
        self.console.print("[dim]--- DECISION: After Logic Check ---[/dim]")
        max_retries = get_config().get("main_pipeline.agent_logic.max_retries", 2)

        if state["regenerate_needed"] and state["retry_count"] <= max_retries:
            label = "first pass" if state["retry_count"] == 0 else "loop"
            self.console.print(f"[dim]    -> Route: 're-generate' ({label})[/dim]")
            return "regenerate"

        if state["regenerate_needed"]:
            self.console.print("[dim]    -> Route: 'clean and grade' (retry limit reached)[/dim]")
        else:
            self.console.print("[dim]    -> Route: 'clean and grade' (answer validated)[/dim]")
        return "proceed"

    # -----------------------------------------------------------------------
    # Solver node  — system prompt separated from user prompt
    # -----------------------------------------------------------------------

    def generate_answer(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Main LLM (Solver)'

        The solver receives:
          system – static Lokad/Envision persona (_SOLVER_SYSTEM_PROMPT)
          user   – the assembled prompt from the agentic workflow
                   (question + verified facts + RAG/grep results + instruction)

        The Mistral model's context is reset before each Solver call so that
        a fresh two-message exchange is used rather than accumulating unrelated
        prior exchanges.
        """
        self.console.print("[dim]--- NODE: Generate Answer (Main LLM) ---[/dim]")
        prompt = state["prompt"]
        prompt += "\n\n### LOKAD ASSISTANT ANSWER:\n"

        if state["verbose"]:
            prompt_panel = Panel(escape(prompt), title="Main LLM Prompt", border_style="purple")

        if self.rate_limit_delay > 0:
            get_collector().record_rate_limit_delay(self.rate_limit_delay)
            time.sleep(self.rate_limit_delay)

        # Reset context so each Solver call is a clean system+user exchange
        self.main_llm.reset_context()
        self.main_llm.append_conversation_history(state.get("previous_qa", []))
        
        get_collector().start_llm_generation("solver")
        generation = self.main_llm.generate_response(
            user_message=prompt,
            system_prompt=_SOLVER_SYSTEM_PROMPT,
        )
        get_collector().end_llm_generation("solver")

        if state["verbose"]:
            generation_panel = Panel(Markdown(generation),
                                     title="LLM Raw Generation", border_style="blue")
            self.console.print(
                Panel(Group(prompt_panel, generation_panel),
                      title=f"Main LLM - {self.main_llm.model_name}", border_style="cyan")
            )

        return {"generation": generation}

    # -----------------------------------------------------------------------
    # Cleaner node  — system prompt separated from user prompt
    # -----------------------------------------------------------------------

    def clean_generated_answer(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Clean Generated Answer'

        The cleaner receives:
          system – static post-processing persona (_CLEANER_SYSTEM_PROMPT),
                   which includes the XML output format instruction.
          user   – the raw generation + the original question for context.

        This replaces the previous approach of baking the XML format
        instruction into the user turn, which is less reliable with
        instruction-tuned models.
        """
        self.console.print("[dim]--- NODE: Clean Generated Answer ---[/dim]")
        raw_generation = state["generation"]

        user_message = (
            f"### QUESTION\n{state['question']}\n\n"
            f"### RAW GENERATION\n{raw_generation}\n\n"
            "### TASK\n"
            "Clean and reformat the Raw Generation into a final answer following "
            "the system instructions exactly."
        )

        content_log = ""
        if state["verbose"]:
            content_log += (
                f"🧹[bold purple] → Cleaning LLM User Message:[/bold purple]\n"
                f"{escape(user_message)}\n"
            )

        if self.rate_limit_delay > 0:
            get_collector().record_rate_limit_delay(self.rate_limit_delay)
            time.sleep(self.rate_limit_delay)

        self.cleaning_llm.reset_context()
        
        get_collector().start_llm_generation("cleaning")
        answer = self.cleaning_llm.generate_response(
            user_message=user_message,
            system_prompt=_CLEANER_SYSTEM_PROMPT,
            temperature = 0.1
        )
        get_collector().end_llm_generation("cleaning")

        answer_match = re.search(
            r"<final_answer>(.*?)</final_answer>", answer,
            re.IGNORECASE | re.DOTALL,
        )
        final_answer = answer_match.group(1).strip() if answer_match else answer.strip()

        if state["verbose"]:
            content_log += (
                f"🧹[bold bright_blue] → Cleaned Final Answer:[/bold bright_blue]\n"
                f"{escape(final_answer)}\n"
            )
            self.console.print(Panel(content_log, title=f"Cleaning - {self.cleaning_llm.model_name}", border_style="green"))

        return {"final_answer": final_answer}

    def validate_answer_sources(self, state: AgentGraphState) -> AgentGraphState:
        """Lightweight non-blocking validation of cited script paths."""
        self.console.print("[dim]--- NODE: Validate Answer Sources ---[/dim]")
        final_answer = state.get("final_answer") or ""

        if not self.answer_validation_cfg.get("enabled", True):
            return {"answer_validation_report": None, "regenerate_needed": False}

        report = self.path_validator.validate_answer(final_answer)
        max_retries = self.answer_validation_cfg.get("max_retries", 3)
        current_retries = state.get("answer_validation_retry_count", 0)

        if not report.get("has_invalid"):
            return {
                "answer_validation_report": report,
                "regenerate_needed": False,
            }

        if current_retries < max_retries:
            feedback = build_validation_feedback(report)
            updated_prompt = (
                f"{state.get('prompt', '')}\n\n### SOURCE PATH VALIDATION FEEDBACK\n"
                f"{feedback}\n"
            )
            self.console.print(
                f"[dim]    -> Invalid cited paths detected. Regenerating answer ({current_retries + 1}/{max_retries}).[/dim]"
            )
            return {
                "answer_validation_report": report,
                "answer_validation_retry_count": current_retries + 1,
                "prompt": updated_prompt,
                "regenerate_needed": True,
                "retry_count": state.get("retry_count", 0) + 1,
            }

        self.console.print("[dim]    -> Invalid cited paths persist. Proceeding with warning section.[/dim]")
        warned_answer = final_answer
        if self.answer_validation_cfg.get("append_warning_section", True):
            warned_answer = append_validation_warning(final_answer, report)
        return {
            "final_answer": warned_answer,
            "answer_validation_report": report,
            "regenerate_needed": False,
        }

    def decide_after_answer_validation(self, state: AgentGraphState) -> str:
        self.console.print("[dim]--- DECISION: After Answer Validation ---[/dim]")
        if state.get("regenerate_needed"):
            self.console.print("[dim]    -> Route: 're-generate' (source validation failed)[/dim]")
            return "regenerate"
        self.console.print("[dim]    -> Route: 'grade_answer' (source validation passed or warning appended)[/dim]")
        return "proceed"

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------

    def build_single_qa_graph(self) -> StateGraph:
        """
        Builds the Advanced Agentic Pipeline.
        Flow: START → Agent → Logic → Generate → Agent … → Clean → Grade → END
        """
        workflow = StateGraph(AgentGraphState)

        workflow.add_node("agentic_workflow", self.run_agentic_workflow)
        workflow.add_node("check_logic", self.check_agent_logic)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("clean_answer", self.clean_generated_answer)
        workflow.add_node("validate_answer_sources", self.validate_answer_sources)
        workflow.add_node("grade_answer", self.grade_answer)

        workflow.add_edge(START, "agentic_workflow")
        workflow.add_edge("agentic_workflow", "check_logic")

        workflow.add_conditional_edges(
            "check_logic",
            self.decide_after_logic_check,
            {
                "regenerate": "generate_answer",
                "proceed": "clean_answer",
            },
        )

        workflow.add_edge("generate_answer", "agentic_workflow")
        # TODO: workflow.add_edge("clean_answer", "validate_answer_sources")
        workflow.add_edge("clean_answer", "grade_answer") # Temporarily bypassing validation for faster iteration; can re-enable after ensuring validation is non-blocking and robust.
        workflow.add_conditional_edges(
            "validate_answer_sources",
            self.decide_after_answer_validation,
            {
                "regenerate": "generate_answer",
                "proceed": "grade_answer",
            },
        )
        workflow.add_edge("grade_answer", END)

        return workflow
