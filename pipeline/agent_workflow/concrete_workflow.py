import config_manager
import time
import re
from langgraph_base import AgentGraphState, ActionLog
from rag.core.base_retriever import RetrievalResult, BaseRetriever
from rag.core.base_embedder import BaseEmbedder
from pipeline.agent_workflow.workflow_base import *
from agents.prepare_agent import prepare_agent

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
            f"{content[:8000]}\n\n" # Safety truncation
            f"### INSTRUCTION\n"
            f"Extract a concise summary from the CONTENT that specifically answers the Query.\n"
            f"If the content is irrelevant to the query, return 'IRRELEVANT'.\n"
            f"Do not add conversational filler."
        )
        
        response = self.llm.generate_response(prompt)
        
        if verbose:
            print(f"💬 → Distillation LLM Prompt:\n{prompt}\n")
            print(f"💬 → Distillation LLM Response:\n{response}\n")
        
        return response

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str, verbose=False) -> List[Tuple[str, str]]:
        """
        Summarize multiple content items in one go and map facts back to their source.
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
            prompt_text += f"--- ITEM {i} ---\n{snippet}\n\n"
            indexed_sources[i] = source

        prompt_text += (
            "### INSTRUCTION\n"
            "Analyze the items above. Extract key facts that help answer the Query.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "You must list every fact followed immediately by its source tag [Source: ITEM X].\n"
            "Example:\n"
            "- The API endpoint is /v1/users [Source: ITEM 0]\n"
            "- The timeout is set to 30s [Source: ITEM 2]\n"
            "\n"
            "Begin list:"
        )

        # 2. Call the LLM
        response = self.llm.generate_response(prompt_text)
        
        # 3. Parse the response using Regex
        distilled_results = []
        
        # Regex explanation:
        # (.*?)             -> Capture the fact text (non-greedy)
        # \[Source:\s*      -> Match literal "[Source:" with optional whitespace
        # ITEM\s*           -> Match literal "ITEM" with optional whitespace
        # (\d+)             -> Capture the Item ID (digits)
        # \]                -> Match literal closing bracket
        pattern = r"(.*?)\s*\[Source:\s*ITEM\s*(\d+)\]"
        
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

        for fact_text, item_id_str in matches:
            try:
                item_id = int(item_id_str)
                
                # Check if the hallucinated ID actually exists in our index
                if item_id in indexed_sources:
                    # Clean up the fact text (remove bullets, newlines, extra spaces)
                    clean_fact = fact_text.strip().lstrip('-').strip()
                    if clean_fact:
                        # Map back to the original file path
                        original_source = indexed_sources[item_id]
                        distilled_results.append((clean_fact, original_source))
                        
            except ValueError:
                continue
        
        if verbose:
            print(f"💬 → Distillation LLM Prompt:\n{prompt_text}\n")
            print(f"💬 → Distillation LLM Response:\n{response}\n")
            print(f"💬 → Parsed Distilled Results:\n{distilled_results}\n")

        return distilled_results

class SimpleRAGTool(BaseRAGTool):
    """
    A simple implementation of BaseRAGTool.
    
    This class uses a provided retriever to perform retrieval operations.
    """
    
    def __init__(self, retriever: BaseRetriever, embedder: BaseEmbedder):
        super().__init__(retriever)
        self.embedder = embedder
    
    def merge_rag_results(self, results):
        k=10
        merged_results = {}
        for result in results:
            if result.chunk.content in merged_results.keys():
                score, chunk, metadata = merged_results[result.chunk.content]
                merged_results[result.chunk.content] = (score + 1/(k+result.rank), chunk, metadata)
                metadata.update(result.chunk.metadata)
            else:
                merged_results[result.chunk.content] = (1/(k+result.rank), result.chunk, result.metadata)
        results_list = sorted(merged_results.items(), key=lambda item: item[1][0], reverse = True)
        return [RetrievalResult(chunk, score, rank+1, metadata) for rank, (_, (score, chunk, metadata)) in enumerate(results_list)]
    
    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks"), verbose = False) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []

        if config_manager.get_config().get('rag.fusion', False):
            base_fusion_question = "Take the following complex question and decompose it into several distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text.\n"
            
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
            raw_questions = self.agent.generate_response(base_fusion_question + query)
            if verbose:
                print(f"Raw answer from LLM for decomposition of the query : {raw_questions}")
            questions = raw_questions.split("$")
            for sub_question in questions:
                emb = self.embedder.embed_text(sub_question)
                results.extend(self.retriever.search(emb, top_k=top_k))
            results = self.merge_rag_results(results)[:top_k]
        else:
            emb = self.embedder.embed_text(query)
            results = self.retriever.search(emb, top_k=top_k)

        return results

class ConcreteAgentWorkflow(BaseAgentWorkflow):
    """
    A concrete implementation of BaseAgentWorkflow for a specific agentic workflow.
    
    This class defines the specific tools and logic used in the agentic workflow.
    """
    
    def __init__(self, rag_tool: BaseRAGTool, grep_tool: BaseGrepTool,
                 script_finder_tool: BaseScriptFinderTool,
                 distillation_tool: BaseDistillationTool):
        super().__init__(rag_tool, grep_tool, script_finder_tool, distillation_tool)
        self.config_manager = config_manager.get_config()
        self.planner_llm = prepare_agent(self.config_manager.get('main_pipeline.agent_logic.planner_llm',
                                                                 self.config_manager.get_default_agent()))
        self.rate_limit_delay = self.config_manager.get('agent.rate_limit_delay', 0)
    
    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        The Planner Node.
        Generates a Plan (Thought) and selects a Tool.
        """
        print("--- SUB-NODE: Agentic Router (Planner) ---")
        
        planning_prompt = self.design_planner_prompt(state)
        
        response = self.planner_llm.generate_response(planning_prompt)
        
        # Parse XML (Simple regex helper)
        import re
        def parse_tag(tag, text):
            match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        thought = parse_tag("thought", response)
        raw_tool = parse_tag("tool", response)
        parameter = parse_tag("parameter", response)

        VALID_TOOLS = {
            "rag_tool", 
            "grep_tool", 
            "script_finder_tool", 
            "simple_regeneration_tool", 
            "grade_answer"
        }


        # Strip whitespace to handle cases like "rag_tool "
        clean_tool = raw_tool.strip()

        if clean_tool in VALID_TOOLS:
            # Happy Path: The tool is valid
            state['tool'] = clean_tool
            state['tool_parameter'] = parameter
        else:
            # Error Path: Redirect to Regeneration (Self-Correction)
            if state["pipeline_state"]["verbose"]:
                print(f"  [System Warning]: Invalid tool '{clean_tool}' detected. redirecting to fix.")
            
            state['tool'] = "simple_regeneration_tool"
            
            # We inject a specific instruction telling the LLM why it failed
            # This becomes the 'tool_parameter' for the regeneration tool
            state['tool_parameter'] = (
                f"SYSTEM ERROR: The tool '{clean_tool}' does not exist. "
                f"You MUST select one of the following valid tools: {', '.join(VALID_TOOLS)}."
            )


        state['current_thought'] = thought
        state["regenerate"] = (state['tool'] != "grade_answer")
        
        if state["pipeline_state"]["verbose"]:
            print("Planner Prompt:")
            print(planning_prompt)
            print(f"\n\nPlanner selected tool: {state['tool']} with parameter: {state['tool_parameter']}")
            print(f"Thought: {thought}")
        
        return state