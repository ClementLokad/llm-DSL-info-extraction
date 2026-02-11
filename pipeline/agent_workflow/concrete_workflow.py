import config_manager
from pipeline.agent_workflow.workflow_base import *
from agents.prepare_agent import prepare_agent
from rag.core.base_parser import BlockType
import time
import re

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
            "   - Parameter: A precise regex pattern (most of the time a simple string suffice). \n"
            "     Optionally restrict scope by adding <sources>PATH_REGEX</sources>. "
            "The results are restricted to files whose path matches the source regex. This allows searching inside specific folders but use ONLY when NECESSARY as you may miss relevant information.\n" 
            "   Optionally restrict scope by adding <block_type>BLOCK_TYPE</block_type> (BLOCK_TYPE must be comma-separated BLOCK_TYPE1,BLOCK_TYPE2...). " 
            "The results are restricted to blocks of the specified type(s). This allows searching for specific code structures but use ONLY when NECESSARY as you may miss relevant information."
            "    BLOCK_TYPE : COMMENT, SECTION_HEADER, IMPORT, READ, WRITE, CONST, EXPORT, TABLE_DEFINITION, ASSIGNMENT, SHOW, KEEP_WHERE, FORM_READ, CONTROL_FLOW, UNKNOWN\n" \
            "   - Example:\n"
            "     • Standard (Simple pattern): <parameter>LotMultiplier</parameter>\n"
            "     • Standard (Complex regex pattern): <parameter>show (linechart|label)</parameter>\n"
            '     • With source filter (Folder scope): <parameter>read "/Manual/Dashboard.ion" <sources>/modules/</sources></parameter>\n'
            '     • With block type filter: <parameter> LotMultiplier <block_type>read</block_type></parameter>\n\n'

            "3. script_finder_tool\n"
            "   - Usage: Read specific files. Use RARELY and only when necessary due to high token cost; use grep_tool with sources instead whenever possible.\n"
            "   - Parameter: Comma-separated filenames or path fragments.\n"
            "   - Example: <parameter>config.nvn, utils/db.nvn</parameter>\n\n"
        )

        # =================================================================
        # MODE 1: KICKOFF (First Pass - Simplified)
        # =================================================================
        if not history:
            prompt = self.base_instructions + (
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
        prompt = self.base_instructions + (
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
            "If YES, your job is done.\n"
            "Note that an unclear answer (e.g., vague statements, lack of sources, or 'I don't know') should be treated as NOT ANSWERED."
            "In this case, if recommendations are given, try to identify the specific missing piece of information and select the tool that can find it.\n\n"
        )
        
        # 4. Verified Facts
        if knowledge_bank:
            prompt += "### VERIFIED FACTS (Assets)\n"
            for i, fact in enumerate(knowledge_bank):
                prompt += f"{i}. {fact}\n"
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
            
            "4. grade_answer\n"
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
    
    def agentic_router(self, state: WorkflowState) -> WorkflowState:
        """
        The Planner Node.
        Generates a Plan (Thought) and selects a Tool.
        """
        self.console.print("[dim]--- SUB-NODE: Agentic Router (Planner) ---[/dim]")
        
        planning_prompt = self.design_planner_prompt(state)
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        if "local_grep_retries" in state and state["pipeline_state"]["execution_history"][-1]["tool"] == "grep_tool":
            base_prompt=" Please answer using the same output format."
            
            len_res = state["local_grep_retries"][1]
            
            state["pipeline_state"]["execution_history"] = state["pipeline_state"]["execution_history"][:-1]
            
            if len_res > get_config().get("main_pipeline.grep_tool.max_results_to_refine"):
                planning_prompt = f"The grep search yielded {len_res} results which is superior to "\
                    f"the limit of {get_config().get('main_pipeline.grep_tool.max_results_to_refine')}, "\
                    "try to slightly narrow down the search or consider using another tool." + base_prompt
                response = self.planner_llm.follow_up_question(planning_prompt)
            elif len_res == 0:
                planning_prompt = f"The grep search yielded no results, "\
                    "try to slightly broaden the search or consider using another tool." + base_prompt
                response = self.planner_llm.follow_up_question(planning_prompt)
            else:
                response = self.planner_llm.generate_response(planning_prompt)
        else:
            response = self.planner_llm.generate_response(planning_prompt)

        # Parse XML (Simple regex helper)
        thought = self._parse_tag("thought", response)
        raw_tool = self._parse_tag("tool", response)
        parameter = self._parse_tag("parameter", response)

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
        state["regenerate"] = (state['tool'] != "grade_answer" and state["pipeline_state"]["retry_count"] <= self.config_manager.get("main_pipeline.agent_logic.max_retries", 5))
        
        if state["pipeline_state"]["verbose"]:
            if "local_grep_retries" in state:
                prompt_content = Text.from_markup(f"Follow-up prompt: {escape(planning_prompt)}\n")
            else:
                prompt_content = Panel(escape(planning_prompt), title="Planner Prompt", border_style="purple")
            tool_content = Text.from_markup(f"\nPlanner selected tool: [bold green]{state['tool']}[/bold green] with "
                                            f"parameter: [bold orange3]{escape(state['tool_parameter'])}[/bold orange3]\n")
            thought_content = Panel(Markdown(thought), title="Planner Thought", border_style="blue")
            self.console.print(Panel(Group(prompt_content, tool_content, thought_content), title="Planner", border_style="bright_red"))
        
        # Distill the previous tool results if they exist
        if "execution_history" in state["pipeline_state"] and len(state["pipeline_state"]["execution_history"]) > 0:
            prev_results = state["pipeline_state"]["execution_history"][-1]["results_to_analyse"]
        else:
            prev_results = None
        if prev_results and len(prev_results) > 0 and state['tool'] != "grade_answer":
            if state["pipeline_state"]["verbose"]:
                self.console.print("[dim]Distilling previous tool results into knowledge bank...[/dim]")
                
            # 2. Batch Distillation
            # Prepare inputs: list of (content, file_path)
            items_to_distill = []
            for r in prev_results:
                # Safely get file path or default to 'Unknown'
                src = r.chunk.metadata.get('original_file_path', 'Unknown Source')
                items_to_distill.append((r.chunk.content, src))

            llm_response = state["pipeline_state"]["generation"]
            
            thought = state["pipeline_state"]["execution_history"][-1]["thought"]
            query = state['pipeline_state']["question"]
            
            # Distill the previous tool results
            new_facts = self.distillation_tool.distill_batch(
                items=items_to_distill,
                query=query,
                thought=thought,
                previous_generation=llm_response,
                verbose=state['pipeline_state']['verbose']
            )
            
            # Add distilled facts to knowledge bank
            if "knowledge_bank" not in state['pipeline_state']:
                state['pipeline_state']["knowledge_bank"] = []
            state['pipeline_state']["knowledge_bank"].extend(new_facts)
            
            # 3. Update History
            outcome_str = f" Extracted {len(new_facts)} relevant facts."
            if state["pipeline_state"]["verbose"]:
                self.console.print(f"[dim]Extracted {len(new_facts)} facts into knowledge bank.[/dim]")
            state["pipeline_state"]["execution_history"][-1]["outcome_summary"] += outcome_str
        
        return state
    
    def use_rag_tool(self, state: WorkflowState) -> WorkflowState:
        """New version that gives the raw output to the LLM as it will
        be distilled at the start of next iteration if necessary"""
        self.console.print("[dim]--- SUB-NODE: RAG Tool ---[/dim]")
        query = state["tool_parameter"]
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.rag_tool.retrieve(query=query, verbose=state['pipeline_state']['verbose'])
        
        if state['pipeline_state']['verbose']:
            self.console.print(f"RAG retrieved {len(results)} results for query: '{escape(query)}'")
        
        # 3. Update History
        count = len(results)
        outcome_str = f"Retrieved {count} chunks."
        self._append_history(state, "rag_tool", query, outcome_str, thought, results)
        
        # Format raw results for the main LLM
        raw_results_str = "\n\n".join([
            f"=== Source: {res.chunk.metadata.get('original_file_path', 'Unknown Source')} ===\n{res.chunk.content}"
            for res in results
        ])
        
        # 4. Prompt for Solver
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### RAG RESULTS\n"
            f"The RAG tool retrieved {len(results)} relevant code chunks for the query '{query}'. Here they are:\n"
            f"{raw_results_str}\n\n"
            f"### INSTRUCTION\n"
            f"Using the RAG results above and all of the previous knowledge, answer the question as best as you can."
        )
        return state

    def use_grep_tool(self, state: WorkflowState) -> WorkflowState:
        """New version that gives the raw output to the LLM as it will
        be distilled at the start of next iteration if necessary"""
        self.console.print("[dim]--- SUB-NODE: Grep Tool ---[/dim]")
        pattern = state["tool_parameter"]
        
        sources_match = re.search(f"<sources>(.*?)</sources>", pattern, re.DOTALL)

        blocktype_match = re.search(f"<block_type>(.*?)</block_type>", pattern, re.DOTALL)
    
        source_regex = None

        block_types = None

        if sources_match:
            # Extract the source regex inside <sources>
            source_regex = sources_match.group(1).strip()
            
            # Remove the entire <sources> block from raw_content to isolate the grep pattern
            # This handles cases like: "read <sources>file1</sources>" -> "read"
            pattern = pattern.replace(sources_match.group(0), "").strip()

        if blocktype_match:
            print(f"Block type filter detected: {blocktype_match.group(1)}")
            block_type_str = blocktype_match.group(1).strip()
            # Support multiple block types separated by commas
            # Convert strings to BlockType Enum objects
            block_types = []
            for bt in block_type_str.split(","):
                bt_clean = bt.strip().upper()
                try:
                    block_types.append(BlockType[bt_clean])
                except KeyError:
                    pass  # Silently ignore invalid block types
            
            # If no valid block types were provided, ignore the filter
            if not block_types:
                block_types = None
            
            # Remove the entire <block_type> block from raw_content to isolate the grep pattern
            pattern = pattern.replace(blocktype_match.group(0), "").strip()
        
        # Retrieve the thought generated by the Planner
        thought = state.get("current_thought", "No reasoning provided.")
        
        # 1. Execute
        results = self.grep_tool.search(pattern=pattern, source_regex=source_regex, bloc_type=block_types)
        
        if not "local_grep_retries" in state:
            state["local_grep_retries"] = (0, len(results))
        else:
            state["local_grep_retries"] = (state["local_grep_retries"][0]+1, len(results))
        
        if state['pipeline_state']['verbose']:
            self.console.print(f"Grep found {len(results)} matches for pattern: '{escape(pattern)}'")
        
        shortened_res = False
        original_result_count = len(results)
        if len(results) > get_config().get("main_pipeline.grep_tool.max_results"):
            results = results[:get_config().get("main_pipeline.grep_tool.max_results")]
            shortened_res = True
        
        if state['pipeline_state']['verbose'] and shortened_res:
                self.console.print(f"Only the first {get_config().get('main_pipeline.grep_tool.max_results')} will be analysed")

        if results == [] and state["local_grep_retries"][0] >= self.config_manager.get("main_pipeline.grep_tool.max_grep_retries", 3):
            new_facts = [f"No matches found for pattern '{pattern}' in {f'sources matching the pattern {source_regex}' if source_regex else 'database'}."]
            if "knowledge_bank" not in state['pipeline_state']:
                state['pipeline_state']["knowledge_bank"] = []
            state['pipeline_state']["knowledge_bank"].extend(new_facts)
        
        # 3. Update History
        outcome_str = f"Grep found {original_result_count} matches."
        if shortened_res:
            outcome_str += f"Only the first {get_config().get('main_pipeline.grep_tool.max_results')} were analyzed."
        self._append_history(state, "grep_tool", pattern, outcome_str, thought, results)

        compacted_results = self.grep_tool.shorten_results(pattern, [res.chunk.content for res in results],
                                                           self.config_manager.get("main_pipeline.grep_tool.max_lines"))
        
        if len(compacted_results) < len(results):
            raise Exception("We lost results in compaction")
        
        # Format raw results for the main LLM
        raw_results_str = "\n\n".join([
            f"=== Source: {res.chunk.metadata.get('original_file_path', 'Unknown Source')} ===\n{compacted_results[i]}"
            for (i, res) in enumerate(results)
        ])
        
        total_sources = set()
        for res in results:
            total_sources.add(res.chunk.metadata.get('original_file_path', 'Unknown Source'))
        
        # 4. Design Prompt for Solver
        base_prompt = self.design_first_part_prompt(state)
        state['rewritten_prompt'] = (
            f"{base_prompt}"
            f"### GREP RESULTS\n"
            
        )
        if shortened_res:
            state['rewritten_prompt'] += f'WARNING: Results truncated, only first {get_config().get("main_pipeline.grep_tool.max_results")} will be given out of {original_result_count}.\n'
        elif results == []:
            state['rewritten_prompt'] += f"WARNING: No matches were found for pattern '{pattern}' in {f'sources matching the pattern {source_regex}' if source_regex else 'database'}.\n"
        else:
            state['rewritten_prompt'] += f"The Grep tool found {len(results)} matches for the pattern '{pattern}' which are from **{len(total_sources)} distinct scripts**."
        state['rewritten_prompt'] += (
            " Here are the results:\n\n"
            f"{raw_results_str}\n\n"
            "### INSTRUCTION\n"
            f"Using the Grep results above and all of the previous knowledge, answer the question as best as you can."
        )
        return state
    
    def refine_grep(self, state: WorkflowState) -> str:
        self.console.print("[dim]--- SUB-DECISION: Grep results validation ---[/dim]")
        num_retries, len_grep_results = state["local_grep_retries"]
        if num_retries >= self.config_manager.get("main_pipeline.grep_tool.max_grep_retries", 3):
            self.console.print(f"[dim]  -> Max grep retries reached, grep results validated[/dim]")
            return "validated"
        
        if 0 == len_grep_results:
            self.console.print(f"[dim]  -> 0 grep results replanning[/dim]")
            return "replan"

        if len_grep_results > self.config_manager.get("main_pipeline.grep_tool.max_results_to_refine"):
            self.console.print(f"[dim]  -> {len_grep_results} grep results which is > {get_config().get('main_pipeline.grep_tool.max_results_to_refine')}, replanning[/dim]")
            return "replan"

        self.console.print(f"[dim]  -> Grep results validated[/dim]")
        return "validated"

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
        
        self.add_conditional_edges(
            "grep_tool",
            self.refine_grep, 
            {
                "replan" : "agentic_router",
                "validated": END
            }
        )
        
        self.add_edge("rag_tool", END)
        self.add_edge("script_finder_tool", END)
        self.add_edge("simple_regeneration_tool", END)
        
        return self.compile()