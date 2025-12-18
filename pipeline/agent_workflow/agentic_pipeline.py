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

class AgenticPipeline(BasePipeline):
    """
    A LangGraph StateGraph that encapsulates an agentic workflow as a sub-graph.
    
    This graph node invokes the agent workflow defined in workflow_base.py,
    allowing for complex planning and tool usage within a larger pipeline.
    """
    
    def __init__(self, agent: BaseAgentWorkflow):
        super().__init__()
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
            print("--- NODE: Agentic Workflow (Subgraph) ---")
            
            # 1. Prepare State for Sub-Graph
            # We must map AgentGraphState to WorkflowState (the TypedDict used in workflow_base)
            sub_input: WorkflowState = {
                "pipeline_state": state, # Pass the full parent state
                "regenerate": False,
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
            return {"retry_count": state["retry_count"] + 1}
        
        # If we get here, the agent is satisfied (or max retries hit elsewhere)
        print("    -> Agent is satisfied. Proceeding to grading.")
        return {"final_answer": state["generation"]}
    
    def decide_after_logic_check(self, state: GraphState) -> str:
        """
        Decision Point: After 'Logic Checker'
        Determines the next step after the logic check.
        - 'if error detected': Returns to 'agentic_workflow' (loop).
        - 'else': Continues to 'clean_generated_answer'.
        """
        print("--- DECISION: After Logic Check ---")
        
        if state["regenerate_needed"] and state["retry_count"] <= get_config().get("main_pipeline.agent_logic.max_retries", 2):
            if state["retry_count"] == 0:
                print("    -> Route: 'generate' (first pass)")
            else:
                print("    -> Route: 're-generate' (loop)")
            return "regenerate"
        else:
            if state["regenerate_needed"]:
                print("    -> Route: clean and grade answer (retry limit reached)")
            else:
                print("    -> Route: clean and grade answer (answer validated)")
            return "proceed"
    
    def generate_answer(self, state):
        print("--- NODE: Generate Answer (Main LLM) ---")
        prompt = state["prompt"]
        
        if state["verbose"]:
            print(f"💬 → LLM Prompt:\n{prompt}\n")
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        generation = self.main_llm.generate_response(prompt)
        
        if state["verbose"]:
            print(f"💬 → LLM RAW Generation:\n{generation}\n")
        
        return {"generation": generation}
    
    def clean_generated_answer(self, state: AgentGraphState) -> AgentGraphState:
        """
        Node: 'Clean Generated Answer'
        Cleans up the raw LLM generation into a final answer format.
        """
        print("--- NODE: Clean Generated Answer ---")
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
        
        if state["verbose"]:
            print(f"🧹 → Cleaning LLM Prompt:\n{prompt}\n")
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        answer = cleaning_llm.generate_response(prompt)
        answer_match = re.search(r"<final_answer>(.*?)</final_answer>", answer, re.IGNORECASE | re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else answer.strip()
        
        if state["verbose"]:
            print(f"🧹 → Cleaned Final Answer:\n{final_answer}\n")
        
        return {"final_answer": final_answer}
    
    def grade_answer(self, state):
        final_answer = state["final_answer"]
        reference_answer = state["reference_answer"]
        
        if state["verbose"]:
            print(f"✅ → Final Answer to Grade:\n{final_answer}\n")
            print(f"💡 → Reference Answer:\n{reference_answer}\n")

        if self.benchmark_type == 'cosine_similarity':
            from pipeline.benchmarks.cosine_sim_benchmark import CosineSimBenchmark 
            print("--- NODE: Cosine Similarity Grade Answer ---")
            
            benchmark = CosineSimBenchmark()
            
            score = benchmark.compute_similarity(final_answer, reference_answer)
            if state["verbose"]:
                print(f"→ Similarity score with '{reference_answer}': {score:.4f}")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}
        
        elif self.benchmark_type == 'llm_as_a_judge':
            from pipeline.benchmarks.llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark
            print("--- NODE: Judge LLM Grade Answer ---")
            
            benchmark = LLMAsAJudgeBenchmark()
            benchmark.initialize()

            #delay to avoid too many requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            score = benchmark.judge(final_answer, reference_answer)
            
            if state["verbose"]:
                print(f"→ LLM Judge score with '{reference_answer}': {score}")
            
            grade = {"score": score,
                    "question": state["question"],
                    "llm_response": state["final_answer"],
                    "reference": state["reference_answer"]}
            
            return {"grade": grade}
    
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
    
    embedder = SentenceTransformerEmbedder(get_config().get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(get_config().get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    index_path = Path("data/faiss_index")
    retriever.load_index(str(index_path))

    rag_tool = SimpleRAGTool(retriever=retriever, embedder=embedder)
    grep_tool = GrepTool()
    script_finder_tool = PathScriptFinder()
    distillation_tool = LLMDistillationTool()
    
    # Pre-compile the agent sub-graph to avoid recompiling on every node call
    agent_workflow = ConcreteAgentWorkflow(
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
            ("Combien de fois /Clean/Items.ion est lu dans /3. Inspectors/5 - Supplier Inspector.nvn?", "")
        ],
        "sub_rag_system": sub_rag_system,
        "grades": [],
        "verbose": True
    }

    print("--- STARTING GRAPH EXECUTION ---")
    final_state = app.invoke(inputs, {"recursion_limit": 100})
    print("--- END OF EXECUTION ---")
    print("\n📊 Résultats du benchmark")
    print("=" * 60)
    for r in final_state["grades"]:
        print(f"Q: {r['question']}")
        if final_state["verbose"]:
            print(f"  Référence : {r['reference']}")
            print(f"  LLM Response: {r['llm_response']}")
        print(f"→ Score: {r['score']:.4f}")
        print("\n" + "-" * 40 + "\n")

    print(f"\nMoyenne globale : {final_state['benchmark_results']['average_score']:.4f}")