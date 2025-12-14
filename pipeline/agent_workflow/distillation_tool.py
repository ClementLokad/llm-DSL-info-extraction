from typing import List, Tuple
import re
from pipeline.agent_workflow.workflow_base import BaseDistillationTool

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
            print(f"💬 → Distillation LLM Prompt:\n{prompt}\n")
            print(f"💬 → Distillation LLM Response:\n{response}\n")
        
        return response

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str, verbose=False) -> List[Tuple[str, str]]:
        """
        Summarize multiple content items in one go and map facts back to their source using XML parsing.
        """
        if not items:
            return []

        # 1. Construct a batch prompt
        prompt_text = (
            f"### CONTEXT\nQuery: {query}\nCurrent Thought: {thought}\n\n"
            f"### DOCUMENTS TO ANALYZE\n"
        )
        
        # We verify sources to map them back later
        indexed_sources = {}
        
        for i, (content, source) in enumerate(items):
            # Limit content length per chunk to avoid context overflow
            snippet = content[:2000] 
            prompt_text += f"--- ITEM {i+1} ---\n{snippet}\n\n"
            indexed_sources[i+1] = source

        prompt_text += (
            "### INSTRUCTION\n"
            "Analyze the items above. Extract key facts that help answer the Query or the Thought.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "\n"
            "### CRITICAL OUTPUT FORMAT\n"
            "You MUST output the results in strict XML format.\n"
            "Wrap each distinct fact in an <entry> tag.\n"
            "Inside <entry>, use <fact> for the content and <source> for the Item ID number.\n"
            "\n"
            "Example:\n"
            "<entry>\n"
            "  <fact>The API endpoint is /v1/users</fact>\n"
            "  <source>0</source>\n"
            "</entry>\n"
            "<entry>\n"
            "  <fact>The timeout is set to 30s</fact>\n"
            "  <source>2</source>\n"
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
                    source_content = source_tag_match.group(1).strip()
                    
                    # ROBUST PARSING: Find the first integer sequence inside the source tag
                    # This handles: "1", "Item 1", "Source: 1", "ID #1" -> all become 1
                    id_match = re.search(r"(\d+)", source_content)
                    
                    if id_match:
                        item_id = int(id_match.group(1))
                        
                        # Map back to the original file path
                        if item_id in indexed_sources and fact_text:
                            original_source = indexed_sources[item_id]
                            distilled_results.append((fact_text, original_source))
                        elif fact_text:
                            distilled_results.append((fact_text, "Unknown Source"))
                    elif fact_text:
                        distilled_results.append((fact_text, "Unknown Source"))
                        
            except (ValueError, AttributeError):
                continue
        
        if verbose:
            print(f"💬 → Distillation LLM Prompt:\n{prompt_text}\n")
            print(f"💬 → Distillation LLM Response:\n{response}\n")
            print(f"💬 → Parsed Distilled Results:\n{distilled_results}\n")

        return distilled_results