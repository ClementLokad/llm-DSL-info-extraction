import config_manager
from pipeline.agent_workflow.workflow_base import *
from agents.prepare_agent import prepare_agent

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
        self.console.print("[dim]--- SUB-NODE: Agentic Router (Planner) ---[/dim]")
        
        planning_prompt = self.design_planner_prompt(state)
        
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
            prompt_content = Panel(planning_prompt, title="Planner Prompt", border_style="purple")
            tool_content = Text.from_markup(f"\nPlanner selected tool: [bold green]{state['tool']}[/bold green] with "
                                            f"parameter: [bold orange3]{state['tool_parameter']}[/bold orange3]\n")
            thought_content = Panel(Markdown(thought), title="Planner Thought", border_style="blue")
            self.console.print(Panel(Group(prompt_content, tool_content, thought_content), title="Planner", border_style="bright_red"))
        
        return state