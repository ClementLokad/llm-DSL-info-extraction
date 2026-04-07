from typing import List, Tuple, Optional
import re
import os
import time
import json
from pipeline.agent_workflow.workflow_base import BaseDistillationTool
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
    "Output only XML <fact> tags — no preamble, no commentary."
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
            time.sleep(self.rate_limit_delay)
 
        self.llm.reset_context()
        response = self.llm.generate_response(
            user_message = user_message,
            system_prompt = _DISTILL_SINGLE_SYSTEM_PROMPT,
            temperature = 0.15
        )
 
        if verbose:
            prompt_content = Panel(escape(user_message), title="Distillation Prompt", border_style="purple")
            response_content = Panel(Markdown(response), title="Distillation Response", border_style="blue")
            self.console.print(Panel(Group(prompt_content, response_content),
                                     title="Distillation Tool", border_style="yellow"))
 
        return response
 
    def distill_batch(
        self,
        items: List[Tuple[str, str, str]],
        query: str,
        thought: str,
        previous_generation: Optional[str] = None,
        verbose=False,
    ) -> List[Tuple[str, List[str]]]:
        """
        Distil multiple content items in one LLM call.
 
        The static Memory Manager persona goes into the system role.
        All dynamic content (query, documents, previous generation) goes into
        the user role so the model receives a clean separation between
        who it is and what it needs to do.
        """
        if not items:
            return []
 
        total_sources = {source for _, _, source in items}
 
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
 
        for i, (content, id, source) in enumerate(items):
            snippet = content[:2000]
            user_message += f"--- ITEM {i+1} (ID: {id}) FROM {source} ---\n{snippet}\n\n"
 
        user_message += (
            "### TASK\n"
            "Analyze the items above. Extract concise but sufficient key facts that help "
            "answer the Query or the Thought.\n"
            "The Agent is moving to the next step and will lose access to these raw documents, "
            "so do not discard key information about the query found here.\n"
            "Discard only irrelevant text. Combine duplicate information.\n"
            "One fact should not be too long — 25 words maximum.\n"
            "For each fact, specify which item IDs (from the --- ITEM headers) contributed to it.\n"
            "\n"
            "### OUTPUT FORMAT\n"
            "Output strictly valid JSON: a list of dictionaries, each with:\n"
            "- \"response\": the extracted fact (string)\n"
            "- \"evidence_ids\": list of item IDs (strings) that support this fact\n"
            "\n"
            "Example:\n"
            "[\n"
            "  {\"response\": \"Overall, 9 files read \\\"Data.ion\\\"\", \"evidence_ids\": [\"0\", \"1\"]},\n"
            "  {\"response\": \"The script /4. Optimization workflow/03.b. Forecasting predicts future demand.\", \"evidence_ids\": [\"2\"]}\n"
            "]\n"
            "\n"
            "Begin JSON output:"
        )
 
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
 
        self.llm.reset_context()
        response = self.llm.generate_response(
            user_message=user_message,
            system_prompt=_DISTILLATION_SYSTEM_PROMPT,
            temperature=0.15
        )
 
        # Parse JSON response
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                raise ValueError("Response is not a list")
            distilled_results = []
            for item in parsed:
                if isinstance(item, dict) and "response" in item and "evidence_ids" in item:
                    fact = item["response"]
                    evidence_ids = item["evidence_ids"] if isinstance(item["evidence_ids"], list) else []
                    distilled_results.append((fact, evidence_ids))
                else:
                    # Skip invalid items
                    continue
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract JSON from markdown or other formats
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    distilled_results = [(item.get("response", ""), item.get("evidence_ids", [])) for item in parsed if isinstance(item, dict)]
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
                                     title="Distillation Tool", border_style="yellow"))
 
        return distilled_results