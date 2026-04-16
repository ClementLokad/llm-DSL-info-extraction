from typing import List, Tuple, Optional
import re
import os
import time
import json
from pipeline.agent_workflow.workflow_base import BaseDistillationTool
from pipeline.stats_collector import get_collector
from get_mapping import get_file_mapping
from rich.console import Group, Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.markup import escape
 
_DISTILLATION_SYSTEM_PROMPT = (
    "You are the Memory Manager of a complex RAG Agent. "
    "Your sole responsibility is to extract and preserve key facts from raw documents "
    "so that the agent can answer its query without re-reading those documents. "
    "Be concise but never discard key information regarding the query. "
    "Output only a valid json string — no preamble, no commentary."
)
 
_DISTILL_SINGLE_SYSTEM_PROMPT = (
    "You are a precise information extractor. "
    "Given a document and a query, extract the facts that help to answer the query. "
    "No preamble, no commentary."
)
 
 
class LLMDistillationTool(BaseDistillationTool):
    """
    Concrete distillation tool backed by an LLM.
 
    Both methods now use generate_response(user_message, system_prompt=...)
    so the static persona lands in the 'system' role and the dynamic content
    (documents, query, previous generation) stays in the 'user' role.
    The distillation LLM context is always reset before each call — distillation
    is stateless by design.
    """
    
    def __init__(self, llm_name: str = None, console: Console = Console()):
        super().__init__(llm_name=llm_name)
        self.mapping = get_file_mapping()
 
    def distill(self, content: str, query: str, thought: str, source: str = None, verbose=False) -> str:
        """Single-item distillation (used by script_finder_tool)."""
 
        user_message = (
            f"### CONTEXT\n"
            f"Query: {query}\n"
            f"Planner Thought: {thought}\n\n"
            f"### CONTENT TO ANALYZE"
        )
        if source:
            user_message += f" FROM {self.mapping.get(os.path.splitext(os.path.basename(source))[0], source)}"
        user_message += (
            f"\n{content[:200_000]}\n\n"   # safety truncation
            f"### TASK\n"
            f"Extract concise information that helps answer the Query or Thought. "
            f"Include relevant context that contribute to understanding the topic."
        )
 
        if self.rate_limit_delay > 0:
            get_collector().record_rate_limit_delay(self.rate_limit_delay)  # Record rate limit delay in stats
            time.sleep(self.rate_limit_delay)
        get_collector().start_llm_generation("distillation")
        self.llm.reset_context()
        response = self.llm.generate_response(
            user_message = user_message,
            system_prompt = _DISTILL_SINGLE_SYSTEM_PROMPT,
            temperature = 0.15
        )
        get_collector().end_llm_generation("distillation")
 
        if verbose:
            prompt_content = Panel(escape(user_message), title="Distillation Prompt", border_style="purple")
            response_content = Panel(Markdown(response), title="Distillation Response", border_style="blue")
            self.console.print(Panel(Group(prompt_content, response_content),
                                     title=f"Distillation Tool - {self.llm.model_name}", border_style="yellow"))
 
        return response
 
    def distill_batch(
        self,
        items: List[Tuple[str, str]],
        query: str,
        thought: str,
        previous_generation: Optional[str] = None,
        verbose=False,
    ) -> List[Tuple[str, List[int]]]:
        """
        Distil multiple content items in one LLM call.
 
        The static Memory Manager persona goes into the system role.
        All dynamic content (query, documents, previous generation) goes into
        the user role so the model receives a clean separation between
        who it is and what it needs to do.
        """
        if not items:
            return []
 
        total_sources = {source for _, source in items}
 
        # --- User message: all dynamic content ---
        user_message = (
            f"### CONTEXT\n"
            f"Query: {query}\n"
            f"Current Thought of the agent: {thought}\n"
        )
 
        if previous_generation:
            user_message += (
                f"\n### PREVIOUS AGENT ATTEMPT\n"
                f"The Main Agent read the documents below and generated this answer:\n"
                f"\"{previous_generation}\"\n"
                f"Use this context to identify which facts were likely used or missed.\n"
            )
 
        user_message += (
            f"\n### DOCUMENTS TO ANALYZE\n"
            f"Here are the {len(items)} items to analyse "
            f"from {len(total_sources)} distinct scripts:\n\n"
        )
 
        for i, (content, source) in enumerate(items):
            snippet = content[:2000]
            user_message += f"--- ITEM {i} FROM {source} ---\n{snippet}\n\n"
 
        user_message += (
            "### TASK\n"
            "Analyze the items above. Extract ONLY the facts that directly and specifically "
            "help answer the Query or the Thought. Be selective — a single precise fact is "
            "better than five vague ones. If two items say the same thing, merge them into "
            "one fact. If an item is irrelevant to the query, ignore it entirely.\n"
            "One fact must be self-contained and concise — 25 words maximum.\n"
            "If the fact references a specific script, function, or file, include its full "
            "path directly in the fact text (do not rely solely on evidence_ids for attribution).\n"
            "\n"
            "### OUTPUT FORMAT\n"
            "Output strictly valid JSON: a list of dictionaries, each with:\n"
            "- \"response\": the extracted fact (string). Include the source path in the text "
            "when the fact is tied to a specific file.\n"
            "- \"evidence_ids\": list of item IDs (integers from the --- ITEM headers) "
            "that support this fact\n"
            "\n"
            "Example:\n"
            "[\n"
            "  {\"response\": \"9 files read 'Data.ion', including /1. utilities/Replenishment.nvn\","
            " \"evidence_ids\": [0, 1]},\n"
            "  {\"response\": \"/4. Optimization workflow/03.b. Forecasting.nvn computes "
            "demand using a Poisson distribution.\", \"evidence_ids\": [2]}\n"
            "]\n"
            "\n"
            "Begin JSON output:"
        )
 
        if self.rate_limit_delay > 0:
            get_collector().record_rate_limit_delay(self.rate_limit_delay)  # Record rate limit delay in stats
            time.sleep(self.rate_limit_delay)
 
        self.llm.reset_context()
        get_collector().start_llm_generation("distillation")
        response = self.llm.generate_response(
            user_message=user_message,
            system_prompt=_DISTILLATION_SYSTEM_PROMPT,
            temperature=0.15
        )
        get_collector().end_llm_generation("distillation")
 
        # Parse JSON response
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                raise ValueError("Response is not a list")
            distilled_results = []
            for item in parsed:
                if isinstance(item, dict) and "response" in item and "evidence_ids" in item:
                    fact = item["response"]
                    evidence_indices = item["evidence_ids"] if isinstance(item["evidence_ids"], list) else []
                    distilled_results.append((fact, evidence_indices))
                else:
                    # Skip invalid items
                    continue
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract JSON from markdown or other formats
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    distilled_results = [(item.get("response", ""), item.get("evidence_ids", [])) 
                                         for item in parsed if isinstance(item, dict)]
                except:
                    distilled_results = []
            else:
                distilled_results = []
 
        if verbose:
            prompt_content = Panel(escape(user_message), title="Distillation Prompt", border_style="purple")
            display_response = response.strip()
            if not display_response.startswith("```"):
                display_response = f"```json\n{display_response}\n```"
            response_content = Panel(Markdown(escape(display_response)),
                                     title="Distillation Response", border_style="blue")
            results_table = Table(title="Parsed Facts", border_style="bold bright_yellow", show_lines=True)
            results_table.add_column("Facts", style="cyan", no_wrap=False)
            results_table.add_column("Evidence IDs", style="yellow", no_wrap=False)
            for fact, ids in distilled_results:
                results_table.add_row(escape(fact), escape(str(ids)))
            self.console.print(Panel(Group(prompt_content, response_content, results_table),
                                     title=f"Distillation Tool - {self.llm.model_name}", border_style="yellow"))
 
        return distilled_results