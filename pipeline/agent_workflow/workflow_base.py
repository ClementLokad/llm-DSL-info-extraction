from typing import TypedDict, List, Optional, Dict, Any, Tuple
from langgraph.graph import END, StateGraph, START
from langgraph_base import GraphState, ActionLog
from rag.core.base_retriever import RetrievalResult
from agents.prepare_agent import *
from config_manager import get_config

# --- 1. New Base Class for Distillation ---

class BaseDistillationTool():
    """
    A base tool for distilling retrieved content into concise knowledge facts.
    Implements 'Distill and Discard' with batch processing to save tokens/calls.
    """
    def __init__(self, llm_name: str = None):
        if llm_name:
            self.llm = prepare_agent(llm_name)
        else:
            self.llm = prepare_agent(get_config().get("main_pipeline.agent_logic.distillation_llm"))

    def distill(self, content: str, query: str, thought: str) -> str:
        """Single item distillation (Legacy/Fallback)."""
        # Placeholder for actual LLM call
        return f"Distilled Fact: Content relevant to '{query}' found."

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str) -> List[Tuple[str, str]]:
        """
        Summarize multiple content items in one go.
        
        Args:
            items: List of (content, source_path_str) tuples.
            query: The user question.
            thought: The planner's current reasoning.
            
        Returns:
            List of (fact_summary, source_path) tuples.
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
            # Limit content length per chunk to avoid context overflow if needed
            snippet = content[:2000] 
            prompt_text += f"--- ITEM {i} (Source: {source}) ---\n{snippet}\n\n"
            indexed_sources[i] = source

        prompt_text += (
            "### INSTRUCTION\n"
            "Analyze the items above. Extract key facts that help answer the Query.\n"
            "Discard irrelevant text. Combine duplicate information.\n"
            "Return the output as a list where each line is: 'Fact [Source: ITEM X]'"
        )

        # 2. Call LLM (Simulated here)
        # response = self.llm.invoke(prompt_text)
        
        # 3. Simulate structured parsing of the LLM response
        # In reality, you would parse the "Source: ITEM X" to recover the file path.
        distilled_results = []
        for i, source in indexed_sources.items():
            # This logic mimics the LLM extracting a fact from Item i
            # and maintaining the correct source path
            fact = f"Distilled info regarding '{query}' extracted from {source.split('/')[-1]}..."
            distilled_results.append((fact, source))
            
        return distilled_results


class BaseGrepTool():
    """A base tool for performing grep-like operations on text data."""
    
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs

    def search(self, pattern: str, case_sensitive: bool = False) -> List[RetrievalResult]:
        """Search for pattern in source files"""
        matches = [] 
        # Implementation of grep logic would go here
        return matches

class BaseScriptFinderTool():
    """A base tool for finding scripts in a codebase."""
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs

    def find_scripts(self, script_names: List[str]) -> List[str]:
        """
        Find scripts by name in source files.
        Returns a list of absolute file paths.
        """
        # Implementation of find logic would go here
        return ["/path/to/found/script.py"]

    def read_file(self, file_path: str) -> str:
        """Helper to read file content."""
        with open(file_path, 'r') as f:
            return f.read()

class BaseRAGTool():
    """A base tool for performing RAG operations."""
    
    def __init__(self, retriever: Any):
        self.retriever = retriever

    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks")) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []
        # Implementation of retrieval logic would go here
        return results

class WorkflowState(TypedDict):
    """State definition for the agent workflow."""
    pipeline_state: GraphState # The state of the overall pipeline
    regenerate: bool 
    error: Optional[str]
    current_thought: Optional[str]
    tool: Optional[str] 
    tool_parameter: Optional[Any] 
    rewritten_prompt: Optional[str]
    # Temporary field to track local history before syncing to pipeline_state
    local_history: Optional[List[ActionLog]]


class BaseAgentWorkflow(StateGraph):
    """A base workflow for agent operations."""
    
    def __init__(self, rag_tool: BaseRAGTool, grep_tool: BaseGrepTool,
                 script_finder_tool: BaseScriptFinderTool,
                 distillation_tool: BaseDistillationTool):
        super().__init__(WorkflowState)
        self.rag_tool = rag_tool
        self.grep_tool = grep_tool
        self.script_finder_tool = script_finder_tool
        self.distillation_tool = distillation_tool
    
    # --- Helper: History Management ---
    
    def _get_history(self, state: WorkflowState) -> List[ActionLog]:
        """Safely retrieve execution history from the pipeline state."""
        # Assuming 'execution_history' is added to GraphState, or we inject it dynamically
        return state['pipeline_state'].get("execution_history", [])

    def _append_history(self, state: WorkflowState, tool: str, param: str, summary: str, thought: str):
        """Append a new action to the history."""
        # Get existing history from the global pipeline state
        history = self._get_history(state)
        
        step_num = len(history) + 1
        
        new_log: ActionLog = {
            "step": step_num,
            "thought": thought,
            "tool": tool,
            "parameter": str(param),
            "outcome_summary": summary
        }
        
        # Ensure the list exists and append
        if "execution_history" not in state['pipeline_state']:
            state['pipeline_state']["execution_history"] = []
        state['pipeline_state']["execution_history"].append(new_log)

    # --- 4. The Context Assembler (Updated for History) ---

    def design_first_part_prompt(self, state: WorkflowState) -> str:
        """
        Constructs the 'World State' prompt.
        Now includes: Question, Knowledge Bank (Facts), AND Execution History (Strategy).
        """
        question = state['pipeline_state']['question']
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        history = self._get_history(state)
        error = state.get('error')
        
        # A. Identity
        prompt = (
            "### SYSTEM ROLE\n"
            "You are an expert technical assistant. \n"
            "Your goal is to answer the user's question by planning and executing data retrieval steps.\n\n"
            f"### QUESTION\n{question}\n\n"
        )

        # B. Execution History (The "Strategy Log")
        # This tells the Planner what has already been tried.
        if history:
            prompt += "### EXECUTION HISTORY (Previous Thoughts & Actions)\n"
            for log in history:
                prompt += (
                    f"Step {log['step']}:\n"
                    f"  - Thought: {log['thought']}\n"  # <--- Now visible to the LLM
                    f"  - Action: {log['tool']}('{log['parameter']}')\n"
                    f"  - Result: {log['outcome_summary']}\n\n"
                )

        # C. Accumulated Knowledge (The "Facts")
        # This tells the Solver (and Planner) what we learned from those actions.
        if knowledge_bank:
            prompt += "### VERIFIED FACTS (Accumulated Knowledge)\n"
            for i, (fact, source) in enumerate(knowledge_bank, 1):
                prompt += f"{i}. {fact} [Source: {source}]\n"
            prompt += "\n"

        # D. Previous Error (for correction)
        if error:
            prompt += "### CRITICAL ERROR IN PREVIOUS LOGIC\n"
            prompt += f"{error}\nAvoid repeating this mistake.\n\n"
        
        return prompt

    # --- 5. The Planner (Consumes History) ---

    def agentic_router(self, state: WorkflowState, verbose=False) -> WorkflowState:
        """
        The Planner Node.
        Generates a Plan (Thought) and selects a Tool.
        """
        if verbose: print("\n- SUB-NODE: Agentic Router (Planner) -")
        
        # 1. Get Context
        base_context = self.design_first_part_prompt(state)
        
        # 2. Planning Prompt with Detailed Tool Definitions
        planning_prompt = (
            f"{base_context}"
            "### TOOL SPECIFICATIONS\n"
            "You have access to the following tools. You must select the one most appropriate for the current step.\n\n"
            
            "1. rag_tool\n"
            "   - Usage: Retrieve general concepts, business logic, or documentation. Use this when you need to understand 'how' or 'why' something works.\n"
            "   - Parameter: A natural language search query.\n"
            "   - Example: <parameter>how is the user authentication flow designed?</parameter>\n\n"
            
            "2. grep_tool\n"
            "   - Usage: Search for exact code patterns, function definitions, or variable names across the codebase. Use this to find specific implementation details.\n"
            "   - Parameter: A specific string pattern or regex to search for.\n"
            "   - Example: <parameter>class UserAuthenticator</parameter>\n\n"
            
            "3. script_finder_tool\n"
            "   - Usage: Locate specific files when you know their approximate names or want to read a specific file found in a previous step.\n"
            "   - Parameter: A comma-separated list of filenames or keywords.\n"
            "   - Example: <parameter>auth.py, login_manager.py</parameter>\n\n"
            
            "4. simple_regeneration_tool\n"
            "   - Usage: Use ONLY if the previous step failed due to a logical error and you want to re-think without using new tools.\n"
            "   - Parameter: A brief instruction on what to correct.\n"
            "   - Example: <parameter>The previous calculation was wrong; re-evaluate using the new facts.</parameter>\n\n"
            
            "5. grade_answer\n"
            "   - Usage: Use this when you have gathered sufficient information in the 'Verified Facts' to answer the User Question.\n"
            "   - Parameter: Type 'None'.\n\n"

            "### PLANNING INSTRUCTION\n"
            "1. Review the 'Execution History' to see what strategies have already been tried.\n"
            "2. Analyze the 'Verified Facts' to see what is missing.\n"
            "3. Formulate a short reasoning plan (Thought).\n"
            "4. Select the next tool and define its parameter based on the specifications above.\n\n"
            
            "### OUTPUT FORMAT\n"
            "You MUST use the following XML format:\n"
            "\n"
            "<tool>[rag_tool | grep_tool | script_finder_tool | simple_regeneration_tool | grade_answer]</tool>\n"
            "<parameter>[The input parameter matching the tool specification]</parameter>"
        )
        
        # 3. Call LLM (Pseudo-code)
        # response = llm.invoke(planning_prompt)
        # For simulation, let's pretend the LLM returns this:
        response_text = """
        <thought>
        I have found the 'main.py' file, but I don't know which port the server listens on.
        I should grep for 'PORT' or 'listen' in that file to find the configuration.
        </thought>
        <tool>grep_tool</tool>
        <parameter>listen</parameter>
        """
        
        # 4. Parse XML (Simple regex helper)
        import re
        def parse_tag(tag, text):
            match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        thought = parse_tag("thought", response_text)
        tool = parse_tag("tool", response_text)
        parameter = parse_tag("parameter", response_text)

        # 5. Update State
        # We store the thought in 'current_thought' so the Tool node can access it later
        state['current_thought'] = thought
        state['tool'] = tool
        state['tool_parameter'] = parameter
        
        state["regenerate"] = (tool != "grade_answer")
        
        return state

    def decide_after_routing(self, state: WorkflowState, verbose=False) -> str:
        if verbose: print("\n- SUB-DECISION: After Routing -")
        
        if state["regenerate"] and state["pipeline_state"]["retry_count"] <= get_config().get("main_pipeline.agent_logic.max_retries", 2):
            if state["tool"] is not None:
                return f"{state['tool']}"
        return "grade_answer"

    # --- 6. The Tools (Producers of History) ---

    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:
        query = state["tool_parameter"]
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.rag_tool.retrieve(query=query)
        
        # 2. Batch Distillation
        # Prepare inputs: list of (content, file_path)
        items_to_distill = []
        for r in results:
            # Safely get file path or default to 'Unknown'
            src = r.metadata.get('original_file_path', 'Unknown Source')
            items_to_distill.append((r.content, src))
            
        # Call batch distillation once
        new_facts = self.distillation_tool.distill_batch(items=items_to_distill, query=query, thought=thought)
        
        if "knowledge_bank" not in state['pipeline_state']:
            state['pipeline_state']["knowledge_bank"] = []
        state['pipeline_state']["knowledge_bank"].extend(new_facts)
        
        # 3. Update History
        count = len(results)
        outcome_str = f"Retrieved {count} chunks. Extracted {len(new_facts)} relevant facts."
        self._append_history(state, "rag_tool", query, outcome_str, thought)
        
        # 4. Prompt for Solver
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
            f"New RAG data regarding '{query}' has been distilled into Verified Facts.\n"
            f"If this is sufficient, answer the question. If not, specify what is missing."
        )
        return state

    def use_grep_tool(self, state: WorkflowState) -> WorkflowState:
        pattern = state["tool_parameter"]
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.grep_tool.search(pattern=pattern)
        
        # 2. Batch Distillation
        # Instead of just listing matches, we analyze the code context around the match
        items_to_distill = []
        for r in results:
            src = r.metadata.get('original_file_path', 'Unknown Source')
            # Pass the code content found by grep
            items_to_distill.append((r.content, src))
            
        # Distill: "What does this code actually do?"
        new_facts = self.distillation_tool.distill_batch(items=items_to_distill, query=pattern, thought=thought)
        
        if "knowledge_bank" not in state['pipeline_state']:
            state['pipeline_state']["knowledge_bank"] = []
        state['pipeline_state']["knowledge_bank"].extend(new_facts)
        
        # 3. Update History
        match_count = len(results)
        outcome_str = f"Grep found {match_count} matches. Code context analyzed and added to Knowledge Bank."
        self._append_history(state, "grep_tool", pattern, outcome_str, thought)
        
        # 4. Design Prompt for Solver
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
            f"Code search for '{pattern}' completed. See results in Verified Facts/History.\n"
            f"If additionnal information is needed, specify what is missing."
            f"Otherwise, analyze the results to answer the question."
        )
        return state

    def use_script_finder_tool(self, state: WorkflowState) -> WorkflowState:
        raw_parameter = state["tool_parameter"] # This is likely a string like "main.py, utils.py"
        thought = state.get("current_thought", "No reasoning provided.")
        
        # --- PARSING FIX ---
        # Convert comma-separated string to list, cleaning whitespace
        if isinstance(raw_parameter, str):
            script_names = [name.strip() for name in raw_parameter.split(',')]
        else:
            script_names = raw_parameter # Fallback if it somehow comes as list
        
        # 1. Execute Find
        # This usually returns a list of paths, e.g., ["/src/utils/cleanup.py"]
        found_paths = self.script_finder_tool.find_scripts(script_names=script_names)
        
        # 2. Read & Distill Content
        items_to_distill = []
        for path in found_paths:
            # We must READ the file content here to make it useful
            # Assuming BaseScriptFinderTool has a helper or we use a file utility
            file_content = self.script_finder_tool.read_file(path)
            items_to_distill.append((file_content, path))
        
        # Now distill the content of the scripts found
        new_facts = self.distillation_tool.distill_batch(items=items_to_distill, query=script_names, thought=thought)
        
        if "knowledge_bank" not in state['pipeline_state']:
            state['pipeline_state']["knowledge_bank"] = []
            
        # Also add the raw location fact, as it's useful
        state['pipeline_state']["knowledge_bank"].append(
            (f"File system confirmed existence of: {found_paths}", "FileSystem")
        )
        # Add the distilled content facts
        state['pipeline_state']["knowledge_bank"].extend(new_facts)
        
        # 3. Update History
        outcome_str = f"Found {len(found_paths)} scripts. Content read and distilled."
        self._append_history(state, "script_finder_tool", script_names, outcome_str, thought)
        
        # 4. Design Prompt
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"The scripts {found_paths} have been located and analyzed, the details have been added to Verified Facts.\n"
            f"If additionnal information is needed, specify what is missing."
            f"Otherwise, analyze the results to answer the question."
        )
        return state

    def use_simple_regeneration_tool(self, state: WorkflowState) -> WorkflowState:
        # Note: Regeneration usually doesn't add to history unless it was a distinct step,
        # but tracking it helps avoid infinite regen loops.
        self._append_history(state, "regeneration", "N/A", "Refined the reasoning prompt.", state["current_thought"])
        additional_advice = state["tool_parameter"]
        
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
            f"The previous attempt resulted in an error. {additional_advice}\n"
            f"Please try again, paying close attention to the previous error."
        )
        return state

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
                "grade_answer": END 
            }
        )
        
        self.add_edge("rag_tool", END)
        self.add_edge("grep_tool", END)
        self.add_edge("script_finder_tool", END)
        self.add_edge("simple_regeneration_tool", END)
        
        return self.compile()