from typing import List, Tuple, Optional
import re
from pipeline.agent_workflow.workflow_base import BaseDistillationTool
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table


class LLMDistillationTool(BaseDistillationTool):
    """
    A concrete implementation of BaseDistillationTool using an LLM.
    
    This class uses a provided LLM to perform text distillation operations.
    """
    
    def distill(self, content: str, query: str, thought: str, verbose=False) -> str:
        """Single item distillation (Legacy/Fallback)."""
        prompt = (
            f"### CONTEXT\n"
            f"Query: {query}\n"
            f"Planner Thought: {thought}\n\n"
            f"### CONTENT TO ANALYZE\n"
            f"{content[:200_000]}\n\n" # Safety truncation
            f"### INSTRUCTION\n"
            f"Extract a concise summary from the CONTENT that specifically answers the Query or the Thought.\n"
            f"If the content is irrelevant, return 'IRRELEVANT'.\n"
            f"Do not add conversational filler."
        )
        
        response = self.llm.generate_response(prompt)
        
        if verbose:
            prompt_content = Panel(prompt, title="Distillation LLM Prompt", border_style="purple")
            response_content = Panel(Markdown(response), title="Distillation LLM Response", border_style="blue")
            self.console.print(Panel(Group(prompt_content, response_content), title="Distillation Tool", border_style="yellow"))
        
        return response

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str, previous_generation: Optional[str] = None, verbose=False) -> List[str]:
        """
        Summarize multiple content items in one go and map facts back to their source using XML parsing.
        Now considers the Main LLM's previous attempt to answer.
        """
        if not items:
            return []

        total_sources = set()
        for (_, source) in items:
            total_sources.add(source)

        # 1. Construct a batch prompt
        prompt_text = (
            f"### SYSTEM ROLE\n"
            f"You are the Memory Manager of a complex RAG Agent. The agent has just attempted to answer a query using the documents below.\n\n"
            f"### CONTEXT\nQuery: {query}\n\nCurrent Thought of the agent: {thought}\n"
        )
        
        if previous_generation:
            prompt_text += (
                f"\n### PREVIOUS AGENT ATTEMPT\n"
                f"The Main Agent read the documents below and generated this answer:\n"
                f"\"{previous_generation}\"\n"
                f"Use this context to identify which facts were likely used or missed.\n\n"
            )

        prompt_text += (
            f"### DOCUMENTS TO ANALYZE\n"
            f"Here are the {len(items)} items to analyse which are from {len(total_sources)} distinct scripts:\n\n"
        )
        
        # We verify sources to map them back later
        indexed_sources = {}
        
        for i, (content, source) in enumerate(items):
            # Limit content length per chunk to avoid context overflow
            snippet = content[:2000] 
            prompt_text += f"--- ITEM {i+1} FROM {source} ---\n{snippet}\n\n"
            indexed_sources[i+1] = source

        prompt_text += (
            "### INSTRUCTION\n"
            "Analyze the items above. Extract concise and yet sufficient key facts that help answer the Query or the Thought.\n"
            "Your goal is to build a persistent memory for the RAG Agent. The Agent is moving to the next step and will lose access "
            "to these raw documents, so you must not discard any important information found here.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "\n"
            "### CRITICAL OUTPUT FORMAT\n"
            "You MUST output the results in strict XML format.\n"
            "Wrap each distinct fact in a <fact> tag.\n"
            "\n"
            "Example 1:\n"
            "<fact>Overall, 9 files read \"Data.ion\"</fact>\n"
            "<fact>4 files read \"Data.ion\" as \"Data[Id unsafe]\"</fact>\n"
            "\n"
            "Example 2:\n"
            "<fact>The script /4. Optimization workflow/03.b. Forecasting tries to predict future demand.</fact>\n"
            "\n"
            "Begin XML output:"
        )

        # 2. Call the LLM
        response = self.llm.generate_response(prompt_text)
        
        # 3. Parse the response using Regex
        fact_pattern = r"<fact>(.*?)</fact>"
        distilled_results = re.findall(fact_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if verbose:
            prompt_content = Panel(prompt_text, title="Distillation LLM Prompt", border_style="purple")
            if not response.strip().startswith("```"):
                response = f"```xml\n{response.strip()}\n```"
            response_content = Panel(Markdown(response), title="Distillation LLM Response", border_style="blue")
            results_content = Table(title="Parsed Distilled Results", border_style="bold bright_yellow", show_lines=True)
            results_content.add_column("Facts", style="cyan", no_wrap=False)
            for fact in distilled_results:
                results_content.add_row(fact)
            self.console.print(Panel(Group(prompt_content, response_content, results_content), title="Distillation Tool", border_style="yellow"))

        return distilled_results