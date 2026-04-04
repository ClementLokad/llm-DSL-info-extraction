from typing import TypedDict, List, Optional, Dict, Any, Tuple
from langgraph.graph import END, StateGraph, START
from rag.core.base_retriever import RetrievalResult
from config_manager import get_config

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# --- 1. Define Graph State ---
# The state is a dictionary that flows through the graph.
# Each node reads this state and writes its results to it.

class GraphState(TypedDict):
    """
    Represents the state of our RAG graph.

    Attributes:
        question: The user's input question.
        reference_answer: The reference answer for benchmarking.
        retrieved_context: The documents retrieved by the RAG node (list of tuples with (content, metadata)).
        prompt: The final prompt sent to the LLM.
        generation: The raw output from the 'Main LLM'.
        final_answer: The answer validated by the 'Logic Checker'.
        regenerate_needed: Boolean indicating if the 'Logic Checker' found an error.
        retry_count: Counter to prevent infinite loops.
        grade: The result from the 'Answer Grader'.
        benchmark_results: The final score from the 'Benchmark'.
        verbose: Whether to print verbose output during processing.
    """
    question: str
    reference_answer: str
    retrieved_context: List[RetrievalResult] # List of documents with metadata
    prompt: str
    generation: str
    final_answer: Optional[str]
    regenerate_needed: bool
    retry_count: int
    grade: Optional[Dict[str, Any]]
    verbose: bool

class ActionLog(TypedDict):
    """Represents a single step in the agent's history."""
    step: int
    thought: str
    tool: str
    parameter: str
    outcome_summary: str # Brief summary of success/failure (not full content)
    results_to_analyse: Optional[List[RetrievalResult]] = None

class KnowledgeElement(TypedDict):
    fact: str
    tool: str
    query: str
    retrieval_results: List[RetrievalResult]


class AgentGraphState(GraphState):
    """
    Represents the state of our Advanced RAG Agent graph.

    Attributes:
        question: The user's input question.
        reference_answer: The reference answer for benchmarking.
        knowledge_bank: List of KnowledgeElements containing facts, tool, query and retrieval results.
        execution_history: List of agent iterations.
        prompt: The final prompt sent to the LLM.
        generation: The raw output from the 'Main LLM'.
        final_answer: The answer validated by the 'Logic Checker'.
        regenerate_needed: Boolean indicating if the 'Logic Checker' found an error.
        retry_count: Counter to prevent infinite loops.
        grade: The result from the 'Answer Grader'.
        benchmark_results: The final score from the 'Benchmark'.
        verbose: Whether to print verbose output during processing.
    """
    question: str
    reference_answer: str
    knowledge_bank: List[KnowledgeElement]
    execution_history: List[ActionLog]
    accumulated_evidence: Dict[str, list[RetrievalResult]]
    prompt: str
    generation: str
    final_answer: Optional[str]
    regenerate_needed: bool
    retry_count: int
    grade: Optional[Dict[str, Any]]
    verbose: bool

class BenchmarkState(TypedDict):
    """
    Represents the state for benchmarking multiple Q/A pairs.

    Attributes:
        qa_pairs: List of (question, reference_answer) pairs.
        grades: List of grading results for each Q/A pair.
        benchmark_results: Aggregated results from the benchmark.
        sub_rag_system: The sub-graph handling individual Q/A processing.
        verbose: Whether to print verbose output during processing.
    """
    qa_pairs: List[Tuple[str, str]]
    grades: List[Dict[str, Any]]
    benchmark_results: Dict[str, Any]
    sub_rag_system: StateGraph
    verbose: bool

class APIError(Exception):
    def __init__(self, message, saved_state):
        super().__init__(message)
        self.saved_state = saved_state


# --- 2. Define Graph Nodes ---
# Each node is a function that takes the state as input
# and returns a dictionary containing the state keys to update.

class BasePipeline:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    def retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Node: 'Retrieval Augmented Generation (RAG)'
        Takes the 'question' from the state and retrieves relevant context.
        (Implies the 'Parser' and 'Data Base' steps upstream).
        """
        self.console.print("[dim]--- NODE: Retrieve Documents ---[/dim]")
        question = state["question"]
        
        # ... Your Retriever logic (e.g., BM25, ChromaDB, etc.) ...
        # retriever = ...
        # documents = retriever.invoke(question)

        retrieved_context = [("Fictional context 1", {}), ("Fictional context 2", {})] # Placeholder
        
        return {"retrieved_context": retrieved_context}

    def engineer_prompt(self, state: GraphState) -> GraphState:
        """
        Node: 'Engineered Prompt'
        Builds the final prompt using the question and the retrieved context.
        Also accounts for past errors if in a correction loop.
        """
        self.console.print("[dim]--- NODE: Engineer Prompt ---[/dim]")
        question = state["question"]
        context = state["retrieved_context"]
        
        # ... Your prompt formatting logic ...
        # template = f"Question: {question}\nContext: {context}\nAnswer:"
        
        prompt = f"Formatted prompt for question: {question}" # Placeholder
        
        return {"prompt": prompt}

    def generate_answer(self, state: GraphState) -> GraphState:
        """
        Node: 'Main LLM'
        Calls the main language model with the prompt.
        """
        self.console.print("[dim]--- NODE: Generate Answer (Main LLM) ---[/dim]")
        prompt = state["prompt"]
        
        # ... Your LLM call logic ...
        # llm = ...
        # generation = llm.invoke(prompt)
        
        generation = "Fictional answer generated by the LLM." # Placeholder
        
        return {"generation": generation}

    def check_logic(self, state: GraphState) -> GraphState:
        """
        Node: 'Logic Checker'
        Verifies the LLM's generation. If an error is found, it flags it.
        Otherwise, it validates the 'final_answer'.
        """
        self.console.print("[dim]--- NODE: Check Logic ---[/dim]")
        generation = state["generation"]
        error_count = state.get("error_count", 0)
        
        # ... Your verification logic (e.g., call another LLM, regex, etc.) ...
        
        if False: # Placeholder error detection logic to avoid overcharging the LLM
            self.console.print("[dim]  ⚠️  -> Error detected. Incrementing counter.[/dim]")
            return {
                "regenerate_needed": True,
                "retry_count": error_count + 1
            }
        else:
            self.console.print("[dim]  ✅  -> No error detected. Validating answer.[/dim]")
            return {
                "regenerate_needed": False,
                "final_answer": generation # The generation is validated
            }

    def grade_answer(self, state: GraphState) -> GraphState:
        """
        Node: 'Answer Grader'
        Compares the 'final_answer' with the 'reference_answer' to produce a score.
        """
        self.console.print("[dim]--- NODE: Grade Answer ---[/dim]")
        final_answer = state["final_answer"]
        reference_answer = state["reference_answer"]
        
        # ... Your grading logic (e.g., call a "judge" LLM, ROUGE score, etc.) ...
        
        grade = {"score": 0.9, "reasoning": "The answer is relevant.",
                 "question": state["question"], "llm_response": final_answer,
                 "reference": reference_answer} # Placeholder
        
        return {"grade": grade}

    def run_qa_pairs(self, state: BenchmarkState) -> BenchmarkState:
        """
        Node: 'Run Q/A Pairs'
        Executes the sub-graph for each Q/A pair and collects grades.
        """
        self.console.print("[dim]--- NODE: Run Q/A Pairs ---[/dim]")
        qa_pairs = state["qa_pairs"]
        sub_rag_system = state["sub_rag_system"]
        
        grades = []
        
        for i, (question, reference_answer) in enumerate(qa_pairs):
            self.console.print(f"-> Processing Q/A pair n°{i+1}/{len(qa_pairs)}:\n  [bold green]Question: {question}[/bold green]\n"
                               f"  [bold purple]Reference Answer: {reference_answer}[/bold purple]")
            # Initialize state for the sub-graph
            sub_state: GraphState = {
                "question": question,
                "reference_answer": reference_answer,
                "retrieved_context": [],
                "prompt": "",
                "generation": "",
                "final_answer": None,
                "regenerate_needed": False,
                "retry_count": 0,
                "grade": None,
                "verbose": state["verbose"]
            }
            
            app = sub_rag_system.compile()
            
            interrupted = False
            
            # Execute the sub-graph
            try:
                final_state = app.invoke(sub_state)
            except KeyboardInterrupt:
                self.console.print(
                    "\n[bold yellow]⚠ Benchmark interrupted by user.[/bold yellow]"
                )
                interrupted = True
            except Exception as exc:
                self.console.print(
                    f"\n[bold red]⚠ Benchmark stopped on question "
                    f"{len(grades)+1}/{len(qa_pairs)} due to error:[/bold red]\n{exc}"
                )
                interrupted = True
            
            if interrupted:
                state["grades"] = grades
                raise APIError("Stopped due to API Error", state)

            # Collect the grade
            grades.append(final_state["grade"])
        
        return {"grades": grades}

    def run_benchmark(self, state: BenchmarkState) -> BenchmarkState:
        """
        Node: 'Benchmark'
        Aggregates grades from multiple Q/A pairs into benchmark results.
        """
        self.console.print("[dim]--- NODE: Benchmark ---[/dim]")
        grades = state["grades"]
        
        # ... Your benchmark aggregation logic ...

        benchmark_results = {"average_score": sum(grade["score"] for grade in grades)
                            / len(grades)} # Placeholder

        return {"benchmark_results": benchmark_results}


    # --- 3. Define Conditional Edges ---

    def decide_after_logic_check(self, state: GraphState) -> str:
        """
        Decision Point: After 'Logic Checker'
        Determines the next step after the logic check.
        - 'if error detected': Returns to 'engineer_prompt' (loop).
        - 'else': Continues to 'grade_answer' (and 'Final Answer' is implicit).
        """
        self.console.print("[dim]--- DECISION: After Logic Check ---[/dim]")
        
        if state["regenerate_needed"] and state["retry_count"] <= get_config().get("main_pipeline.agent_logic.max_retries", 2):
            self.console.print("[dim]    -> Route: 're-generate' (loop)[/dim]")
            return "regenerate"
        else:
            if state["regenerate_needed"]:
                self.console.print("[dim]    -> Route: 'grade_answer' (retry limit reached)[/dim]")
            else:
                self.console.print("[dim]    -> Route: 'grade_answer' (answer validated)[/dim]")
            return "proceed"

    # --- 4. Assemble the Graph ---

    def build_single_qa_graph(self) -> StateGraph:
        """
        Builds a simplified graph for single Q/A without benchmarking.
        """

        # Initialize the graph
        workflow = StateGraph(GraphState)

        # Add nodes to the graph
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("engineer_prompt", self.engineer_prompt)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("check_logic", self.check_logic)
        workflow.add_node("grade_answer", self.grade_answer)

        # Define the entry point
        # The flow starts with document retrieval
        workflow.add_edge(START, "retrieve_documents")

        # Add edges (connections)
        workflow.add_edge("retrieve_documents", "engineer_prompt")
        workflow.add_edge("engineer_prompt", "generate_answer")
        workflow.add_edge("generate_answer", "check_logic")

        # Add the conditional edge
        workflow.add_conditional_edges(
            "check_logic",  # Source node
            self.decide_after_logic_check, # Decision function
            {
                # 'if error detected' (route name) -> 'generate_answer' (target node)
                "regenerate": "generate_answer",
                # 'else' (route name) -> 'grade_answer' (target node)
                "proceed": "grade_answer"
            }
        )

        # The 'Grader' is the end of the flow
        workflow.add_edge("grade_answer", END)

        return workflow

    def build_full_benchmark_graph(self) -> StateGraph:
        """
        Builds the complete RAG graph with all nodes and edges.
        """

        # Initialize the graph
        workflow = StateGraph(BenchmarkState)

        # Add nodes to the graph
        workflow.add_node("run_qa_pairs", self.run_qa_pairs)
        workflow.add_node("benchmark", self.run_benchmark)

        # Define the entry point
        workflow.add_edge(START, "run_qa_pairs")

        # Add edges (connections)
        workflow.add_edge("run_qa_pairs", "benchmark")
        workflow.add_edge("benchmark", END)

        return workflow



# --- 5. Compile and Run ---

if __name__ == "__main__":
    pipeline = BasePipeline()
    
    # Build the sub-graph for single Q/A processing
    sub_rag_system = pipeline.build_single_qa_graph()
    
    # Build the full benchmark graph
    workflow = pipeline.build_full_benchmark_graph()

    # Compile the graph
    app = workflow.compile()
    
    # Inject the sub-graph into the benchmark graph's initial state
    inputs = {
        "qa_pairs": [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("Who wrote '1984'?", "'1984' was written by George Orwell."),
            ("List the primary colors.", "The primary colors are red, blue, and yellow.")
        ],
        "sub_rag_system": sub_rag_system,
    }

    # Execute the graph
    print("--- STARTING GRAPH EXECUTION ---")

    final_state = app.invoke(inputs)
    print(final_state["benchmark_results"])
    
    print("--- END OF EXECUTION ---") 