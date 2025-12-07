import sys
import os

# Add the specific folder to sys.path
# '..' means go up one level, then into 'my_modules'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langgraph.graph import END, StateGraph, START
from config_manager import get_config
from pipeline.agent_workflow.workflow_base import *
from langgraph_base import AgentGraphState, BasePipeline

class AgenticPipeline(BasePipeline):
    """
    A LangGraph StateGraph that encapsulates an agentic workflow as a sub-graph.
    
    This graph node invokes the agent workflow defined in workflow_base.py,
    allowing for complex planning and tool usage within a larger pipeline.
    """
    
    def __init__(self, agent: BaseAgentWorkflow):
        super().__init__()
        self.agent = agent.build_graph()
        
    def run_agentic_workflow(self, state: AgentGraphState) -> AgentGraphState:
            """
            Node: 'Agentic Workflow'
            Invokes the sub-graph defined in workflow_base.py.
            """
            print("--- NODE: Agentic Workflow (Subgraph) ---")
            
            # 1. Prepare State for Sub-Graph
            # We must map AgentGraphState to WorkflowState (the TypedDict used in workflow_base)
            sub_input: WorkflowState = {
                "pipeline_state": state, # Pass the full parent state
                "regenerate": state.get("regenerate_needed", False),
                "error": None, 
                "tool": None,
                "tool_parameter": None,
                "rewritten_prompt": None,
                "current_thought": None,
                "local_history": None
            }
            
            # 2. Run Sub-Graph
            # This runs the Planner -> Tool loop until it finishes one 'turn'
            final_sub_state = self.agent.invoke(sub_input)
            
            # 3. Extract Results back to Main State
            updated_pipeline_state = final_sub_state["pipeline_state"]
            
            # The agent sub-graph creates a 'rewritten_prompt' designed for the Solver
            new_prompt = final_sub_state.get("rewritten_prompt")
            if not new_prompt:
                # Fallback if agent just started or didn't update prompt
                new_prompt = f"Question: {state['question']}"

            return {
                "knowledge_bank": updated_pipeline_state.get("knowledge_bank"),
                "execution_history": updated_pipeline_state.get("execution_history"),
                "prompt": new_prompt,
                "regenerate_needed": final_sub_state["regenerate"],
                "retry_count": state["retry_count"]
            }

    def check_agent_logic(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Logic Checker (Agentic)'
        Decides if the loop should continue based on the Agent's actions.
        """
        print("--- NODE: Check Agent Logic ---")
        
        # 1. Check if this is the first pass (No generation yet)
        if not state.get("generation"):
            print("    -> First pass detected. Forcing generation.")
            return {"regenerate_needed": True}
        
        # 2. Check if the Agent requested regeneration
        if state["regenerate_needed"]:
            print(f"    -> Agent used tool '{state['execution_history'][-1]['tool']}'. Regenerating answer.")
            return state
        
        # If we get here, the agent is satisfied (or max retries hit elsewhere)
        print("    -> Agent is satisfied. Proceeding to grading.")
        return {
            "final_answer": state["generation"]
        }
    
    def build_agentic_qa_graph(self) -> StateGraph:
        """
        Builds the Advanced Agentic Pipeline.
        Flow: START -> Agent -> Logic -> Generate -> Agent ... -> Grade -> END
        """
        # Initialize with the specific Agent state
        workflow = StateGraph(AgentGraphState)

        # Add Nodes
        workflow.add_node("agentic_workflow", self.run_agentic_workflow)
        workflow.add_node("check_logic", self.check_agent_logic)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("grade_answer", self.grade_answer)

        # 1. START -> agentic_workflow
        # The Agent analyzes state and prepares the prompt/tools
        workflow.add_edge(START, "agentic_workflow")

        # 2. agentic_workflow -> check_logic
        # Logic checker decides if we need to generate an answer (first pass or update)
        # or if we are done.
        workflow.add_edge("agentic_workflow", "check_logic")

        # 3. check_logic -> (Conditional)
        workflow.add_conditional_edges(
            "check_logic",
            self.decide_after_logic_check,
            {
                # If regenerate needed (first pass or new info found):
                "regenerate": "generate_answer",
                # If Agent is satisfied:
                "proceed": "grade_answer"
            }
        )

        # 4. generate_answer -> agentic_workflow
        # After generating a draft/answer, we loop back to the agent 
        # so it can critique it or find more info based on it.
        workflow.add_edge("generate_answer", "agentic_workflow")

        # 5. grade_answer -> END
        workflow.add_edge("grade_answer", END)

        return workflow

if __name__ == "__main__":
    # Initialize tools for the agentic workflow (Mocks/Placeholders)
    # In a real app, these would be injected with actual clients
    rag_tool = BaseRAGTool(retriever=None)
    grep_tool = BaseGrepTool(search_dirs=[])
    script_finder_tool = BaseScriptFinderTool(search_dirs=[])
    distillation_tool = BaseDistillationTool()
    
    # Pre-compile the agent sub-graph to avoid recompiling on every node call
    agent_workflow = BaseAgentWorkflow(
        rag_tool, 
        grep_tool, 
        script_finder_tool, 
        distillation_tool
    )
    pipeline = AgenticPipeline(agent_workflow)
    
    print(">>> Building AGENTIC Pipeline")
    sub_rag_system = pipeline.build_agentic_qa_graph()
    
    workflow = pipeline.build_full_benchmark_graph()
    app = workflow.compile()
    
    inputs = {
        "qa_pairs": [
            ("What is the capital of France?", "Paris is the capital of France.")
        ],
        "sub_rag_system": sub_rag_system,
        "grades": [],
        "verbose": False
    }

    print("--- STARTING GRAPH EXECUTION ---")
    final_state = app.invoke(inputs, {"recursion_limit": 100})
    print(final_state["benchmark_results"])
    print("--- END OF EXECUTION ---")