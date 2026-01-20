from typing import List, Tuple
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

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str, verbose=False) -> List[Tuple[str, str]]:
        """
        Summarize multiple content items in one go and map facts back to their source using XML parsing.
        """
        if not items:
            return []

        total_sources = set()
        for (_, source) in items:
            total_sources.add(source)

        # 1. Construct a batch prompt
        prompt_text = (
            f"### SYSTEM ROLE\n"
            f"You are the assistant of a complex RAG Agent. The agent has to answer the query that follows.\n\n"
            f"### CONTEXT\nQuery: {query}\nCurrent Thought of the agent: {thought}\n\n"
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
            "Your goal is thus to help build a persistent memory for the RAG Agent. The Agent will not have access to the "
            "sources you are analyzing so you must not discard any important information.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "\n"
            "### CRITICAL OUTPUT FORMAT\n"
            "You MUST output the results in strict XML format.\n"
            "Wrap each distinct fact in an <entry> tag.\n"
            "Inside <entry>, use <fact> for the content and <source> for the Item ID number (or comma-separated list of IDs).\n"
            "\n"
            "Example 1:\n"
            "<entry>\n"
            "  <fact>Overall, 9 files read \"Data.ion\"</fact>\n"
            "  <source>1,2,3,4,5,7,8,9,10</source>\n"
            "</entry>\n"
            "<entry>\n"
            "  <fact>4 files read \"Data.ion\" as \"Data[Id unsafe]\"</fact>\n"
            "  <source>1,2,5,7</source>\n"
            "</entry>\n"
            "\n"
            "Example 2:\n"
            "<entry>\n"
            "  <fact>The script /4. Optimization workflow/03.b. Forecasting tries to predict future demand.</fact>\n"
            "  <source>3</source>\n"
            "</entry>\n"
            "\n"
            "Begin XML output:"
        )

        # 2. Call the LLM
        response = self.llm.generate_response(prompt_text)
        
        # 3. Parse the response using Regex
        distilled_results = []
        
        # Step A: Find all entry blocks
        entry_pattern = r"<entry>(.*?)</entry>"
        entries = re.findall(entry_pattern, response, re.IGNORECASE | re.DOTALL)
        
        # Step B: Parse inside each entry
        for entry_block in entries:
            try:
                # Extract the fact text
                fact_match = re.search(r"<fact>(.*?)</fact>", entry_block, re.IGNORECASE | re.DOTALL)
                
                # Extract the source tag content (everything between tags)
                source_tag_match = re.search(r"<source>(.*?)</source>", entry_block, re.IGNORECASE | re.DOTALL)
                
                if fact_match and source_tag_match:
                    fact_text = fact_match.group(1).strip()
                    source_content = source_tag_match.group(1).strip().split(",")
                    
                    # ROBUST PARSING: Find the first integer sequence inside the source tag
                    # This handles: "1", "Item 1", "Source: 1", "ID #1" -> all become 1
                    id_matches = [re.search(r"(\d+)", source) for source in source_content]
                    sources = set()
                    
                    for id_match in id_matches:
                        if id_match:
                            item_id = int(id_match.group(1))
                            
                            # Map back to the original file path
                            if item_id in indexed_sources and fact_text:
                                original_source = indexed_sources[item_id]
                                sources.add(original_source)
                            elif fact_text:
                                sources.add("Unknown Source")
                        elif fact_text:
                            sources.add("Unknown Source")
                    distilled_results.append((fact_text, ", ".join(list(sources))))
                        
            except (ValueError, AttributeError):
                continue
        
        if verbose:
            prompt_content = Panel(prompt_text, title="Distillation LLM Prompt", border_style="purple")
            if not response.strip().startswith("```"):
                response = f"```xml\n{response.strip()}\n```"
            response_content = Panel(Markdown(response), title="Distillation LLM Response", border_style="blue")
            results_content = Table(title="Parsed Distilled Results", border_style="bold bright_yellow", show_lines=True)
            results_content.add_column("Fact", style="cyan", no_wrap=False)
            results_content.add_column("Sources", style="magenta", no_wrap=False)
            for fact, sources in distilled_results:
                results_content.add_row(fact, sources.replace(", ", ",\n"))
            self.console.print(Panel(Group(prompt_content, response_content, results_content), title="Distillation Tool", border_style="yellow"))

        return distilled_results