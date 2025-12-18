import re
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from langgraph.graph import END, StateGraph, START
from langgraph_base import AgentGraphState, ActionLog
from rag.core.base_retriever import RetrievalResult, BaseRetriever
from rag.core.base_embedder import BaseEmbedder
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

    def distill(self, content: str, query: str, thought: str, verbose=False) -> str:
        """Single item distillation (Legacy/Fallback)."""
        # Placeholder for actual LLM call
        return f"Distilled Fact: Content relevant to '{query}' found."

    def distill_batch(self, items: List[Tuple[str, str]], query: str, thought: str, verbose=False) -> List[Tuple[str, str]]:
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

    def search(self, pattern: str, sources: Optional[List[str]] = None) -> List[RetrievalResult]:
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
        return ["/path/to/found/script.nvn"]

    def read_file(self, file_path: str) -> str:
        """Helper to read file content."""
        with open(file_path, 'r') as f:
            return f.read()

class BaseRAGTool():
    """A base tool for performing RAG operations."""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k = get_config().get("rag.top_k_chunks")) -> List[RetrievalResult]:
        """Retrieve relevant documents based on the query"""
        results = []
        # Implementation of retrieval logic would go here
        return results

class WorkflowState(TypedDict):
    """State definition for the agent workflow."""
    pipeline_state: AgentGraphState # The state of the overall pipeline
    regenerate: bool 
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

    def _parse_tag(self, tag, text):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    # --- 4. The Context Assembler (Updated for History) ---

    def design_first_part_prompt(self, state: WorkflowState) -> str:
        """
        Constructs the 'World State' prompt.
        Now includes: Question, Knowledge Bank (Facts), AND Execution History (Strategy).
        """
        question = state['pipeline_state']['question']
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        history = self._get_history(state)
        thought = state.get('current_thought', None)
        
        # A. Identity
        prompt = (
            "### SYSTEM ROLE\n"
            "You are an expert technical assistant working on a **Lokad Envision** codebase.\n"
            "Lokad is a supply chain optimization company, and Envision is their specialized programming language designed for quantitative supply chain logic and probabilistic forecasting.\n"
            "Your goal is to answer the user's question by exploiting data from previous retrieval steps.\n\n"
            f"### QUESTION\n{question}\n\n"
        )

        # B. Accumulated Knowledge (The "Facts")
        # This tells the Solver (and Planner) what we learned from those actions.
        if knowledge_bank:
            prompt += "### VERIFIED FACTS (Accumulated Knowledge)\n"
            for i, (fact, source) in enumerate(knowledge_bank, 1):
                prompt += f"{i}. {fact} [Source: {source}]\n"
            prompt += "\n"
        else:
            prompt += "### VERIFIED FACTS\n(No relevant facts have been gathered yet.)\n\n"

        # C. Current thought (for correction)
        if thought:
            prompt += "### CURRENT THOUGHT\n"
            prompt += f"{thought}\n\n"
        
        return prompt
    
    def _get_optimized_history_str(self, history: List[ActionLog]) -> str:
        """
        Génère une chaîne de caractères optimisée pour l'historique.
        Stratégie : Garder complet le début et la fin, résumer le milieu.
        """
        if not history:
            return "(No previous actions taken.)"
            
        total_steps = len(history)
        history_str = ""
        
        # Seuil : Si moins de 5 étapes, on affiche tout
        if total_steps <= 5:
            for log in history:
                history_str += (
                    f"- Step {log['step']}:\n"
                    f"  * Thought: {log['thought']}\n"
                    f"  * Tool: {log['tool']} -> {log['parameter']}\n"
                    f"  * Outcome: {log['outcome_summary']}\n"
                )
            return history_str

        # Sinon : Compression
        # 1. Première étape (Contexte initial)
        first = history[0]
        history_str += (
            f"- Step {first['step']} (Start):\n"
            f"  * Thought: {first['thought']}\n"
            f"  * Tool: {first['tool']} -> {first['parameter']}\n"
            f"  * Outcome: {first['outcome_summary']}\n"
        )
        
        # 2. Résumé du milieu (Texte statique pour économiser des tokens, pas d'appel LLM)
        history_str += (
            f"\n... [Steps 2 to {total_steps - 3} were executed. Details hidden for brevity. "
            f"The agent tried various strategies which led to the current state.] ...\n\n"
        )
        
        # 3. Les 3 dernières étapes (Détail complet pour la prise de décision immédiate)
        for log in history[-3:]:
            history_str += (
                f"- Step {log['step']}:\n"
                f"  * Thought: {log['thought']}\n"
                f"  * Tool: {log['tool']} -> {log['parameter']}\n"
                f"  * Outcome: {log['outcome_summary']}\n"
            )
            
        return history_str

    def design_planner_prompt(self, state: WorkflowState) -> str:
        """
        Creates a specialized prompt for the 'Planner' role.
        Handles two modes:
        1. Kickoff Mode: First pass, simplified, focus on initial discovery.
        2. Review Mode: Subsequent passes, full context analysis, focus on gap filling.
        """
        question = state['pipeline_state']['question']
        history = self._get_history(state)
        
        first_3_tools_desc = (
            "1. rag_tool\n"
            "   - Usage: Retrieve Envision concepts or Lokad business logic.\n"
            "   - Parameter: A natural language query describing the concept to find.\n"
            "   - Example: <parameter>how does the refund policy work?</parameter>\n\n"

            "2. grep_tool\n"
            "   - Usage: Find specific Envision code implementations, variable definitions, or error strings.\n"
            "   - Parameter: A precise regex or string pattern. Optionally restrict scope by adding <sources>...</sources>"
            " but use ONLY when NECESSARY as you may miss relevant information.\n"
            "   - Example:\n"
            "     • Standard : <parameter>show linechart</parameter>\n"
            "     • With source filter : <parameter>read \"/Manual/Dashboard.ion\" <sources>forecasting.nvn, income.nvn</sources></parameter>\n\n"

            "3. script_finder_tool\n"
            "   - Usage: Read specific files. Use RARELY and only when necessary due to high token cost; use grep_tool with sources instead whenever possible.\n"
            "   - Parameter: Comma-separated filenames or path fragments.\n"
            "   - Example: <parameter>config.nvn, utils/db.nvn</parameter>\n\n"
        )

        # =================================================================
        # MODE 1: KICKOFF (First Pass - Simplified)
        # =================================================================
        if not history:
            prompt = (
                "### SYSTEM ROLE\n"
                "You are the **Strategic Planner** for an advanced RAG agent working on a **Lokad Envision** codebase.\n"
                "Lokad is a supply chain optimization company, and Envision is their specialized programming language designed for quantitative supply chain logic and probabilistic forecasting.\n"
                "You are initiating a new investigation. Your job is to determine the best FIRST step to gather information.\n\n"
                
                f"### MISSION GOAL\n{question}\n\n"
                
                "### AVAILABLE TOOLS & SPECIFICATIONS\n"
                "Select the tool best suited to start the investigation.\n\n"
                
                ) + first_3_tools_desc + (

                "### PLANNING INSTRUCTIONS\n"
                "1. Analyze the 'Mission Goal'. Identify the most critical keyword or concept.\n"
                "2. Determine if this requires high-level documentation (RAG) or low-level code search (Grep/Script).\n"
                "3. Select the tool and define a precise parameter.\n\n"
                
                "### OUTPUT FORMAT\n"
                "Respond strictly in this XML format:\n"
                "<thought>\n"
                "[Explain your reasoning. What is the first piece of info needed?]\n"
                "</thought>\n"
                "<tool>[rag_tool | grep_tool | script_finder_tool]</tool>\n"
                "<parameter>[Your precise input parameter]</parameter>"
            )
            return prompt

        # =================================================================
        # MODE 2: CONTINUATION (Subsequent Passes - Full Context)
        # =================================================================
        
        knowledge_bank = state['pipeline_state'].get("knowledge_bank", [])
        # We safely get generation, defaulting to "No output yet" if None
        previous_generation = state['pipeline_state'].get('generation') or "(No generation available)"
        
        # 1. System Role: The Strategist
        prompt = (
            "### SYSTEM ROLE\n"
            "You are the **Investigation Supervisor** for an advanced RAG agent working on a **Lokad Envision** codebase.\n"
            "Lokad is a supply chain optimization company, and Envision is their specialized programming language designed for quantitative supply chain logic and probabilistic forecasting.\n"
            "Your primary goal is EFFICIENCY. You must decide if the current information is sufficient to answer the question.\n"
            "If the answer is found, you MUST stop the investigation immediately.\n\n"
        )

        # 2. The Mission (Question)
        prompt += f"### MISSION GOAL\n{question}\n\n"
        

        # 3. PROPOSED SOLUTION (Critical Context)
        prompt += (
            "### PROPOSED SOLUTION (From Main Agent)\n"
            "The Main Agent has reviewed the facts and generated this answer:\n"
            f"\"{previous_generation}\"\n\n"
            "**CRITICAL CHECK**: Does this proposed solution directly and fully answer the Mission Goal? "
            "If YES, your job is done.\n\n"
        )
        
        # 4. Verified Facts
        if knowledge_bank:
            prompt += "### VERIFIED FACTS (Assets)\n"
            for i, (fact, source) in enumerate(knowledge_bank, 1):
                prompt += f"{i}. {fact} [Source: {source}]\n"
        else:
            prompt += "### VERIFIED FACTS\n(Knowledge bank is empty.)\n"
        prompt += "\n"

        # 5. Investigation History
        prompt += "### HISTORY (Previous Steps)\n"
        prompt += self._get_optimized_history_str(history)
        prompt += "\n"

        # 6. Tool Specifications (Full Menu)
        prompt += (
            "### AVAILABLE TOOLS & SPECIFICATIONS\n"
            "Select the most precise tool for the current need.\n\n"
            
            ) + first_3_tools_desc + (
            
            "4. simple_regeneration_tool\n"
            "   - Usage: Use ONLY if the previous step failed due to a logical error and you want to re-think without using new tools.\n"
            "   - Parameter: A brief instruction on what to correct.\n"
            "   - Example: <parameter>The previous calculation was wrong; re-evaluate using the new facts.</parameter>\n\n"
            
            "5. grade_answer\n"
            "   - Usage: If the current answer is satisfying, use this tool to finalize the process.\n"
            "   - Parameter: Type 'None'.\n\n"
        )

        # 7. Decision Algorithm (Logic Flow)
        # CHANGED: Priority #1 is checking for completion.
        prompt += (
            "### DECISION LOGIC (Follow Strictly)\n"
            "1. **COMPLETION CHECK**: Read the 'PROPOSED SOLUTION'. Does it answer the 'Mission Goal'?\n"
            "   - YES -> STOP. Select <tool>grade_answer</tool>.\n"
            "   - NO -> Proceed to Step 2.\n\n"
            
#            "2. **REDUNDANCY CHECK**: Do NOT search for 'confirmation' or 'corroboration' if the facts are already clear.\n"
#            "   - If you are just double-checking -> STOP. Select <tool>grade_answer</tool>.\n\n"
            
            "2. **GAP ANALYSIS**: If the answer is genuinely missing or unsatisfactory (e.g., 'I don't know' or 'File not found'), select the tool to find that specific missing piece.\n"
            "   - Look at the 'History'. Have we already tried the obvious step? If yes, try a different angle (e.g., if grep failed, try RAG).\n"
            "   - Be specific in your parameter choice. Vague parameters yield vague results.\n\n"
            
            "### OUTPUT FORMAT\n"
            "Respond strictly in this XML format:\n"
            "<thought>\n"
            "[ANSWER FOUND | Your strategic reasoning in which you define the missing information and why this tool will find it.]\n"
            "</thought>\n"
            "<tool>[rag_tool | grep_tool | script_finder_tool | simple_regeneration_tool | grade_answer]</tool>\n"
            "<parameter>[Your precise input parameter]</parameter>"
        )

        return prompt

    # --- 5. The Planner (Consumes History) ---

    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        The Planner Node.
        Generates a Plan (Thought) and selects a Tool.
        """
        print("--- SUB-NODE: Agentic Router (Planner) ---")
        
        planning_prompt = self.design_planner_prompt(state)
        
        # Call LLM (Pseudo-code)
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
        
        # Parse XML (Simple regex helper)
        thought = self._parse_tag("thought", response_text)
        tool = self._parse_tag("tool", response_text)
        # TODO: Map tool name to actual tool if needed
        parameter = self._parse_tag("parameter", response_text)

        # 5. Update State
        # We store the thought in 'current_thought' so the Tool node can access it later
        state['current_thought'] = thought
        state['tool'] = tool
        state['tool_parameter'] = parameter
        
        state["regenerate"] = (tool != "grade_answer" and state["pipeline_state"]["retry_count"] <= get_config().get("main_pipeline.agent_logic.max_retries", 5))
        
        if state["pipeline_state"]["verbose"]:
            print("Planner Prompt:")
            print(planning_prompt)
            print(f"\n\nPlanner selected tool: {tool} with parameter: {parameter}")
            print(f"Thought: {thought}")
        
        return state

    def decide_after_routing(self, state: WorkflowState) -> str:
        print("--- SUB-DECISION: After Routing ---")
        
        if state["regenerate"]:
            if state["pipeline_state"]["verbose"]:
                print(f"    -> Routing to tool: {state['tool']}")
            return f"{state['tool']}"
        return "grade_answer"

    # --- 6. The Tools (Producers of History) ---

    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:    
        print("--- SUB-NODE: RAG Tool ---")
        query = state["tool_parameter"]
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.rag_tool.retrieve(query=query, verbose=state['pipeline_state']['verbose'])
        
        if state['pipeline_state']['verbose']:
            print(f"RAG retrieved {len(results)} results for query: '{query}'")
        
        # 2. Batch Distillation
        # Prepare inputs: list of (content, file_path)
        items_to_distill = []
        for r in results:
            # Safely get file path or default to 'Unknown'
            src = r.chunk.metadata.get('original_file_path', 'Unknown Source')
            items_to_distill.append((r.chunk.content, src))
            
        # Call batch distillation once
        new_facts = self.distillation_tool.distill_batch(items=items_to_distill, query=query,
                                                         thought=thought, verbose=state['pipeline_state']['verbose'])
        
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
        print("--- SUB-NODE: Grep Tool ---")
        pattern = state["tool_parameter"]
        
        sources_match = re.search(f"<sources>(.*?)</sources>", pattern, re.DOTALL)
    
        sources = None

        if sources_match:
            # Extract the CSV string inside <sources>
            sources_str = sources_match.group(1)
            # Convert to a clean list of filenames
            sources = [s.strip() for s in sources_str.split(',') if s.strip()]
            
            # Remove the entire <sources> block from raw_content to isolate the grep pattern
            # This handles cases like: "read <sources>file1</sources>" -> "read"
            pattern = pattern.replace(sources_match.group(0), "").strip()
        
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.grep_tool.search(pattern=pattern, sources=sources)
        
        shortened_res = False
        if len(results) > get_config().get("main_pipeline.grep_tool.max_results"):
            results = results[:get_config().get("main_pipeline.agent_logic.max_results")]
            shortened_res = True
        
        
        if state['pipeline_state']['verbose']:
            print(f"Grep found {len(results)} matches for pattern: '{pattern}'")
        
        # 2. Batch Distillation
        # Instead of just listing matches, we analyze the code context around the match
        items_to_distill = []
        for r in results:
            src = r.chunk.metadata.get('original_file_path', 'Unknown Source')
            # Pass the code content found by grep
            items_to_distill.append((r.chunk.content, src))
            
        # Distill: "What does this code actually do?"
        new_facts = self.distillation_tool.distill_batch(items=items_to_distill, query=pattern,
                                                         thought=thought, verbose=state['pipeline_state']['verbose'])
        
        if results == []:
            new_facts.append((f"No matches found for pattern '{pattern}' in {', '.join(sources) if sources else 'All Sources'}.",
                              ", ".join(sources) if sources else "All Sources"))
        
        if "knowledge_bank" not in state['pipeline_state']:
            state['pipeline_state']["knowledge_bank"] = []
        state['pipeline_state']["knowledge_bank"].extend(new_facts)
        
        # 3. Update History
        match_count = len(results)
        outcome_str = f"Grep found {match_count} matches."
        if shortened_res:
            outcome_str += f"Only the first {get_config().get('main_pipeline.grep_tool.max_results')} were analyzed."
        outcome_str += f"Extracted {len(new_facts)} relevant facts."
        self._append_history(state, "grep_tool", pattern, outcome_str, thought)
        
        # 4. Design Prompt for Solver
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            "### INSTRUCTION\n"
            f"Code search for '{pattern}' completed. "
        )
        if shortened_res:
            state['rewritten_prompt'] += f'\nWARNING: Results truncated, only first {get_config().get("main_pipeline.grep_tool.max_results")} were analyzed.\n'
        elif results == []:
            state['rewritten_prompt'] += f"\nWARNING: No matches were found for pattern '{pattern}' in {', '.join(sources) if sources else 'All Sources'}.\n"
        state['rewritten_prompt'] += (
            "See results in Verified Facts/History.\n"
            "If additional information is needed, specify what is missing. Otherwise, analyze the results to answer the question.\n"
        )
        return state

    def use_script_finder_tool(self, state: WorkflowState) -> WorkflowState:
        print("--- SUB-NODE: Script Finder Tool ---")
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
        new_facts = []
        for path in found_paths:
            # We must READ the file content here to make it useful
            # Assuming BaseScriptFinderTool has a helper or we use a file utility
            file_content = self.script_finder_tool.read_file(path)
            new_facts.append((self.distillation_tool.distill(content=file_content,
                                                            query=state["pipeline_state"]["question"],
                                                            thought=thought, verbose=state['pipeline_state']['verbose']), path))
        
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
            f"### INSTRUCTION\n"
            f"The scripts {found_paths} have been located and analyzed, the details have been added to Verified Facts.\n"
            f"If additional information is needed, specify what is missing. Otherwise, analyze the results to answer the question.\n"
        )
        return state

    def use_simple_regeneration_tool(self, state: WorkflowState) -> WorkflowState:
        print("--- SUB-NODE: Simple Regeneration Tool ---")
        # Note: Regeneration usually doesn't add to history unless it was a distinct step,
        # but tracking it helps avoid infinite regen loops.
        self._append_history(state, "regeneration", "N/A", "Refined the reasoning prompt.", state["current_thought"])
        additional_advice = state["tool_parameter"]
        
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### INSTRUCTION\n"
            f"The previous attempt resulted in an unsatisfactory answer.\n{additional_advice}\n"
            f"Please try again, paying close attention to the previous answer."
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