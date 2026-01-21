import sys
import time
import re
from pathlib import Path

# Add the specific folder to sys.path
# '..' means go up one level, then into 'my_modules'
sys.path.append(str(Path(__file__).parent.parent.parent))

from langgraph.graph import END, StateGraph, START
from config_manager import get_config
from pipeline.agent_workflow.workflow_base import *
from langgraph_base import AgentGraphState, BasePipeline, GraphState
from pipeline.agent_workflow.concrete_workflow import ConcreteAgentWorkflow
from pipeline.agent_workflow.distillation_tool import LLMDistillationTool
from pipeline.agent_workflow.grep_tool import GrepTool
from pipeline.agent_workflow.script_finder_tool import PathScriptFinder
from pipeline.agent_workflow.rag_tool import SimpleRAGTool
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

class AgenticPipeline(BasePipeline):
    """
    A LangGraph StateGraph that encapsulates an agentic workflow as a sub-graph.
    
    This graph node invokes the agent workflow defined in workflow_base.py,
    allowing for complex planning and tool usage within a larger pipeline.
    """

    def __init__(self, console: Console, agent: BaseAgentWorkflow):
        super().__init__(console)
        self.main_llm = prepare_agent(get_config().get('main_pipeline.agent_logic.main_llm',
                                      get_config().get_default_agent()))
        self.rate_limit_delay = get_config().get('agent.rate_limit_delay', 0)
        self.agent = agent.build_graph()
        self.benchmark_type= get_config().get_benchmark_type()
        
    def run_agentic_workflow(self, state: AgentGraphState) -> AgentGraphState:
            """
            Node: 'Agentic Workflow'
            Invokes the sub-graph defined in workflow_base.py.
            """
            self.console.print("[dim]--- NODE: Agentic Workflow (Subgraph) ---[/dim]")
            
            # 1. Prepare State for Sub-Graph
            # We must map AgentGraphState to WorkflowState (the TypedDict used in workflow_base)
            sub_input: WorkflowState = {
                "pipeline_state": state, # Pass the full parent state
                "regenerate": False,
                "error": None, 
                "tool": None,
                "tool_parameter": None,
                "rewritten_prompt": None,
                "current_thought": None
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
                "knowledge_bank": updated_pipeline_state.get("knowledge_bank", []),
                "execution_history": updated_pipeline_state.get("execution_history", []),
                "prompt": new_prompt,
                "regenerate_needed": final_sub_state["regenerate"],
                "retry_count": state["retry_count"]
            }

    def check_agent_logic(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Logic Checker (Agentic)'
        Decides if the loop should continue based on the Agent's actions.
        """
        self.console.print("[dim]--- NODE: Check Agent Logic ---[/dim]")
        
        # 1. Check if this is the first pass (No generation yet)
        if not state.get("generation"):
            self.console.print("[dim]    -> First pass detected. Forcing generation.[/dim]")
            return {"regenerate_needed": True}
        
        # 2. Check if the Agent requested regeneration
        if state["regenerate_needed"]:
            self.console.print(f"[dim]    -> Agent used tool '{state['execution_history'][-1]['tool']}'. Regenerating answer.[/dim]")
            return {"retry_count": state["retry_count"] + 1}
        
        # If we get here, the agent is satisfied (or max retries hit elsewhere)
        self.console.print("[dim]    -> Agent is satisfied. Proceeding to grading.[/dim]")
        return {"final_answer": state["generation"]}
    
    def decide_after_logic_check(self, state: GraphState) -> str:
        """
        Decision Point: After 'Logic Checker'
        Determines the next step after the logic check.
        - 'if error detected': Returns to 'agentic_workflow' (loop).
        - 'else': Continues to 'clean_generated_answer'.
        """
        self.console.print("[dim]--- DECISION: After Logic Check ---[/dim]")
        
        if state["regenerate_needed"] and state["retry_count"] <= get_config().get("main_pipeline.agent_logic.max_retries", 2):
            if state["retry_count"] == 0:
                self.console.print("[dim]    -> Route: 'generate' (first pass)[/dim]")
            else:
                self.console.print("[dim]    -> Route: 're-generate' (loop)[/dim]")
            return "regenerate"
        else:
            if state["regenerate_needed"]:
                self.console.print("[dim]    -> Route: 'clean and grade answer' (retry limit reached)[/dim]")
            else:
                self.console.print("[dim]    -> Route: 'clean and grade answer' (answer validated)[/dim]")
            return "proceed"
    
    def generate_answer(self, state):
        self.console.print("[dim]--- NODE: Generate Answer (Main LLM) ---[/dim]")
        prompt = state["prompt"]
        if state["verbose"]:
            prompt_content = Panel(prompt, title="Main LLM Prompt", border_style="purple")
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        generation = self.main_llm.generate_response(prompt)
        
        if state["verbose"]:
            generation_content = Panel(Markdown(generation), title="LLM Raw Generation", border_style="blue")
            self.console.print(Panel(Group(prompt_content, generation_content), title="Main LLM", border_style="cyan"))
        
        return {"generation": generation}
    
    def clean_generated_answer(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Clean Generated Answer'
        Cleans up the raw LLM generation into a final answer format.
        """
        self.console.print("[dim]--- NODE: Clean Generated Answer ---")
        raw_generation = state["generation"]
        
        cleaning_llm = prepare_agent(get_config().get('main_pipeline.agent_logic.cleaning_llm',
                                     get_config().get_default_agent()))
        
        prompt = (
            "### INSTRUCTION\n"
            "Clean and format the following LLM-generated answer into a concise final answer.\n"
            "Remove any extraneous information, tool usage notes, or internal thoughts.\n"
            "Do NOT add ANY conversational filler.\n\n"
            f"### QUESTION\n{state['question']}\n\n"
            "### OUTPUT FORMAT\n"
            "Respond strictly in this XML format:\n"
            "<final_answer>[The final answer]</final_answer>\n\n"
            f"### RAW GENERATION\n{raw_generation}\n"
        )
        
        content = ""
        
        if state["verbose"]:
            content += f"🧹[bold purple] → Cleaning LLM Prompt:[/bold purple]\n{prompt}\n"
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        answer = cleaning_llm.generate_response(prompt)
        answer_match = re.search(r"<final_answer>(.*?)</final_answer>", answer, re.IGNORECASE | re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else answer.strip()
        
        if state["verbose"]:
            content += f"🧹[bold bright_blue] → Cleaned Final Answer:[/bold bright_blue]\n{final_answer}\n"
            self.console.print(Panel(content, title="Cleaning", border_style="green"))
        
        return {"final_answer": final_answer}
    
    def build_single_qa_graph(self) -> StateGraph:
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
        workflow.add_node("clean_answer", self.clean_generated_answer)
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
                "proceed": "clean_answer"
            }
        )

        # 4. generate_answer -> agentic_workflow
        # After generating a draft/answer, we loop back to the agent 
        # so it can critique it or find more info based on it.
        workflow.add_edge("generate_answer", "agentic_workflow")

        # 5. clean_answer -> grade_answer
        workflow.add_edge("clean_answer", "grade_answer")

        # 6. grade_answer -> END
        workflow.add_edge("grade_answer", END)

        return workflow

if __name__ == "__main__":
    
    console = Console()
    embedder = SentenceTransformerEmbedder(get_config().get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(get_config().get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    index_path = Path("data/faiss_index")
    retriever.load_index(str(index_path))

    rag_tool = SimpleRAGTool(retriever=retriever, embedder=embedder)
    grep_tool = GrepTool()
    script_finder_tool = PathScriptFinder()
    distillation_tool = LLMDistillationTool(console=console)
    
    # Pre-compile the agent sub-graph to avoid recompiling on every node call
    agent_workflow = ConcreteAgentWorkflow(
        rag_tool, 
        grep_tool, 
        script_finder_tool, 
        distillation_tool
    )
    pipeline = AgenticPipeline(console, agent_workflow)
    
    console.print("[dim]>>> Building AGENTIC Pipeline[/dim]")
    sub_rag_system = pipeline.build_single_qa_graph()
    
    workflow = pipeline.build_full_benchmark_graph()
    app = workflow.compile()
    
    inputs = {
        "qa_pairs": [
            ("Existe-t-il un endroit où retrouver des informations condensées pour analyser les différents fournisseurs ?", "27")
        ],
        "sub_rag_system": sub_rag_system,
        "verbose": True
    }

    console.print("[dim]--- STARTING GRAPH EXECUTION ---[/dim]")
    final_state = app.invoke(inputs, {"recursion_limit": 100})
    console.print("[dim]--- END OF EXECUTION ---[/dim]")
        
    console.print(Markdown("# Benchmark Results"))
        
    table = Table(title="Benchmark Grades", show_lines=True)
    table.add_column("Question", style="cyan", no_wrap=False)
    table.add_column("Score", style="magenta")
    
    for r in final_state["grades"]:
        console.print(f"[bold green]Question: {r['question']} [/bold green]\n")
        table.add_row(r['question'], f"{r['score']:.4f}")
        if final_state["verbose"]:
            console.print(f"[bold purple]  Référence: {r['reference']}[/bold purple]")
            console.print("\n[bold blue]  LLM: [/bold blue]")
            console.print(Markdown(f"{r['llm_response']}"))

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]Moyenne globale : {final_state['benchmark_results']['average_score']:.4f}[/bold]")