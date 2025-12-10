#!/usr/bin/env python3
"""Oracle-driven entry point that orchestrates preprocess + thought graph reasoning."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from agents.semantic_agent import SemanticAgent, SemanticAgentInput
from agents.syntax_agent import SyntaxAgent, SyntaxAgentInput
from agents.code_agent import CodeAgent, CodeAgentInput
from llms import LLMClient, prepare_llm
from llms.exceptions import LLMCommunicationError
from question_preprocess import (
    LLMQuestionClarifier,
    LLMQuestionDecomposer,
    LLMRouter,
    QuestionClarifierInput,
    QuestionDecomposerInput,
    RouterInput,
)
from utils import load_config, load_mapping, parse_llm_json, render_ascii_tree


# Load .env files so local CLI runs automatically pick up provider secrets.
load_dotenv()


COMMON_ORACLE_PRIMER = (
        """
Oracle operates on mirrored Envision DSL (``.nvn``) dashboards stored under ``env_scripts/`` plus
their ``mapping.txt`` index. Every script is deterministic: it ingests Ion datasets under
``/Clean``/``/Input`` via literal verbs and renders dashboards through explicit ``show`` blocks.

Mini tutorial for Envision verbs (never confuse them):
- ``import "/path" as Alias`` pulls helper modules (colors, parameters). It does NOT touch data
    files, so imports must not be counted as reads/writes.
- ``read "/Clean/Foo.ion" as Table`` loads Ion datasets into an in-memory table. Optional
    ``expect[...]`` clauses enforce keys, and schema columns follow inside ``with`` blocks.
- ``write "/Output/..." as Table`` or ``export "/Output/..."`` persist computed tables. They are
    state-changing operations distinct from reads.
- ``show table|summary|scalar|markdown|tabs|barchart|logo`` directives only drive UI tiles; they do
    not alter data.
- Control verbs such as ``keep``, ``where``, ``group by``, ``update``, ``delete``, ``when``, and
    ``with`` shape the dataset already loaded in memory.

General rules:
- Treat quoted paths as case-sensitive literals. Example:
    ``read "/Clean/Items.ion" as Items[Id unsafe] with ...``
- Separate verb counts accurately: imports reference modules, reads/writes/exports operate on Ion
    files, and ``show`` blocks render dashboards.
- Mirror the user's verb (read vs write vs import vs export) when drafting sub-questions or
    planner nodes; never rephrase an ``import`` request into ``read`` or ``write`` counts, and honor
    the router justification verbatim when it cites a specific action.
- Stick to deterministic evidence surfaced via scanners or grep-style commands; never invent
    scripts beyond the mirrored repository.
"""
).strip()


DEFAULT_SCOPE_PROMPT = (
        COMMON_ORACLE_PRIMER
        + """
You are {identity}, known internally as "Oracle". You receive user questions and must decide
whether the message is a greeting, relevant to the Envision DSL investigation program, or out of scope.
Return strictly valid JSON with the following schema:
{{
  "status": "greeting" | "project" | "off_topic",
  "reply": "short natural language answer in the user's language",
  "reason": "concise justification"
}}
Guidelines:
- Treat any polite hello/thanks as a greeting and introduce yourself as Oracle.
- "Project" applies to anything touching Envision NVN scripts, deterministic tooling (grep, FAISS, index builds),
  or business questions traceable to those scripts.
- "Off_topic" if the user requests poetry, unrelated coding help, or any topic outside the current workspace scope.
- Always mention the allowed project scope when refusing a request.
- Always respond in the same language as the user.
User message: {question}
Shared context: {context}
Project scope reminder: {project_scope}
"""
).strip()


DEFAULT_THOUGHT_PROMPT = (
    COMMON_ORACLE_PRIMER
    + """
You are the "Thought Graph Planner" assisting Oracle. After clarifier/decomposer/router analysis,
you decide how to explore the workspace using semantic and syntax agents. Return strictly valid JSON:
{{
  "decision": "continue" | "stop",
  "reason": "<=30 words explaining the choice",
  "nodes": [
    {{
      "id": "thought-<int>",
      "parent": "root" | "<existing node id>",
      "action": "semantic" | "syntax",
      "question": "query to pass to the downstream agent",
      "context": "optional short context",
      "hints": ["optional", "hints"],
      "max_results": optional integer,
      "notes": "<=25 words justification"
    }}
  ]
}}
Constraints:
- Never output more than {max_branches} nodes per response.
- Maximum depth is {max_depth}; parent depth information is given via history.
- Only use actions from this list: {allowed_actions}.
- Pending router actions must be scheduled first: {router_actions}.
- Node IDs must be unique and cannot reuse: {used_ids}.
- Entire search is capped at {max_total_steps} executed nodes, so stay focused.
- When evidence feels sufficient, set decision to "stop" and emit an empty node list.
- Mirror the user's language when drafting questions and notes.
Context for this planning round:
- User question: {question}
- Normalized question: {normalized_question}
- User context: {context}
- Clarifier output: {clarifier}
- Decomposition output: {decomposition}
- Router verdict: {router}
- Executed steps so far: {history}
"""
).strip()


DEFAULT_SYNTHESIZER_PROMPT = (
    COMMON_ORACLE_PRIMER
    + """
You are Oracle's "Synthesizer". Combine deterministic evidence from Envision scripts
and thinking loop results to craft an actionable explanation. Always mirror the user's language.

## CRITICAL: Process Termination Status
The thinking loop terminated with: **{termination_status}**
- "completed" = Thinker reached satisfactory confidence → Results are reliable
- "limit_reached" = A limit was hit (iterations/steps/depth) → Results may be incomplete
- "forced_stop" = Emergency stop → Results are partial

When termination_status is "limit_reached" or "forced_stop":
1. Evaluate if the collected evidence is sufficient to answer the question
2. If YES: answer normally but mention "l'exploration a été limitée, mais les résultats semblent suffisants"
3. If NO: clearly state what's missing and suggest next steps ("pour une réponse complète, il faudrait...")

## Response Format
Return strictly valid JSON:
{{
    "answer": "final narrative (<= 180 words)",
    "confidence": 0.0-1.0,
    "is_complete": true/false,
    "needs_more_investigation": "null or explanation of what's missing",
    "highlights": ["optional bullet summaries included only when helpful"]
}}

## Inputs
- Question: {question}
- Router intent: {intent}
- User context: {context}
- Termination status: {termination_status}
- Termination reason: {termination_reason}
- Thinking loop confidence: {thinking_confidence}
- Inline evidence JSON (<= {max_inline_refs} entries): {inline_evidence}
- Evidence count: {evidence_count}
- Overflow count (entries kept for appendix): {overflow_count}
- Agent stats (semantic/syntax calls): {agent_stats}
- Appendix directive ("inline" or "appendix"): {appendix_flag}

## Rules
- Never invent scripts beyond the evidence provided.
- When inline evidence is empty, admit that no deterministic snippet was found and suggest next
  investigative steps grounded in the workspace constraints.
- When {appendix_flag} == "appendix", reference the appendix explicitly.
- The first sentence must state the count of results found.
- Set is_complete=false if termination_status != "completed" AND evidence seems insufficient.
- Confidence must reflect BOTH evidence strength AND process completeness.
"""
).strip()


ROUTER_ACTION_MAP = {
    LLMRouter.SYNTAX_ACTION: "syntax",
    LLMRouter.SEMANTIC_ACTION: "semantic",
}


@dataclass
class ScopeDecision:
    """Structured verdict returned by the scope classifier."""

    status: str
    reply: str
    reason: str


@dataclass
class StageExecution:
    """Trace describing the outcome of one preprocess stage."""

    name: str
    success: bool
    payload: Optional[Dict[str, Any]]
    justification: str = ""
    attempts: int = 1


@dataclass
class PreprocessBundle:
    """Aggregated state produced by the clarifier/decomposer/router stack."""

    normalized_question: str
    stages: List[StageExecution] = field(default_factory=list)
    clarifier: Optional[Dict[str, Any]] = None
    decomposition: Optional[Dict[str, Any]] = None
    router: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            "normalized_question": self.normalized_question,
            "clarifier": self.clarifier,
            "decomposition": self.decomposition,
            "router": self.router,
            "stages": [asdict(stage) for stage in self.stages],
        }


@dataclass
class ThoughtStepRequest:
    """Planner directive instructing Oracle to trigger a downstream worker."""

    node_id: str
    parent_id: str
    action: str
    question: str
    context: Optional[str] = None
    hints: List[str] = field(default_factory=list)
    max_results: Optional[int] = None
    justification: str = ""


@dataclass
class ThoughtStepResult:
    """Outcome of executing a planner directive."""

    node_id: str
    parent_id: str
    action: str
    depth: int
    success: bool
    summary: str
    output: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    attempts: int = 1
    display_id: str = ""
    is_retry: bool = False  # True if this was a retry of a previous action

    def serialize(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "action": self.action,
            "depth": self.depth,
            "success": self.success,
            "summary": self.summary,
            "output": self.output,
            "justification": self.justification,
            "attempts": self.attempts,
            "display_id": self.display_id,
        }


@dataclass
class ThoughtPlan:
    """Planner response controlling the next expansion of the thought graph."""

    decision: str
    reason: str
    steps: List[ThoughtStepRequest] = field(default_factory=list)


@dataclass
class ThoughtLimits:
    """Guardrails applied to the hierarchical search."""

    max_iterations: int = 3
    min_iterations: int = 1
    max_branches: int = 2
    max_depth: int = 3
    max_total_steps: int = 6


# =============================================================================
# GRAPH OF THOUGHTS - Core data structures
# =============================================================================

@dataclass
class ThoughtNode:
    """A node in the Graph of Thoughts.
    
    Indexing follows hierarchical pattern:
    - Root children: 1, 2, 3, ... (up to max_branches)
    - Children of node 1: 11, 12, 13, ...
    - Children of node 12: 121, 122, 123, ...
    
    Each node represents ONE agent call (semantic, syntax, code, etc.)
    """
    index: str                    # "1", "11", "121", etc.
    parent_index: str             # "" for root children, parent's index otherwise
    agent: str                    # "semantic", "syntax", "code"
    description: str              # Brief explanation of what the agent did
    question: str                 # The question/task passed to the agent
    
    # Execution state
    success: bool = False
    attempts: int = 1             # Number of tries (retries count here, not in iterations)
    output: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    
    # Tree structure
    children: List[str] = field(default_factory=list)  # Indices of child nodes
    depth: int = 0                # Distance from root (len(index) essentially)
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "parent_index": self.parent_index,
            "agent": self.agent,
            "description": self.description,
            "attempts": self.attempts,
            "success": self.success,
            "summary": self.summary,
            "children": self.children,
            "depth": self.depth,
        }


@dataclass  
class SplitPoint:
    """Tracks a point where exploration branched into parallel paths.
    
    When the Thinker decides to explore multiple alternatives (e.g., 
    'vendeur' AND 'seller'), it creates a split. We simulate parallelism
    via DFS: explore first branch deeply, then backtrack to this split
    and explore the next branch.
    """
    parent_index: str             # The node index BEFORE the split
    branches: List[str]           # Indices of parallel branches (e.g., ["11", "12", "13"])
    explored_branches: List[str]  # Branches already fully explored
    pending_branches: List[str]   # Branches not yet explored
    created_at_iteration: int     # When this split was created
    
    def all_explored(self) -> bool:
        """Check if all branches have been explored."""
        return len(self.pending_branches) == 0
    
    def next_branch(self) -> Optional[str]:
        """Get the next branch to explore (or None if done)."""
        if self.pending_branches:
            return self.pending_branches[0]
        return None
    
    def mark_explored(self, branch_index: str):
        """Mark a branch as fully explored."""
        if branch_index in self.pending_branches:
            self.pending_branches.remove(branch_index)
            self.explored_branches.append(branch_index)


@dataclass
class ThinkingStats:
    """Statistics about the thinking process."""
    iterations: int = 0           # Number of reasoning cycles (Thinker decisions)
    total_steps: int = 0          # Total agent calls including retries
    max_depth_reached: int = 0    # Deepest node visited
    agents_called: Dict[str, int] = field(default_factory=dict)  # Count per agent
    retries: int = 0              # Number of retry attempts
    splits_created: int = 0       # Number of parallel splits
    branches_explored: int = 0    # Number of branches fully explored
    
    def record_agent_call(self, agent: str, attempts: int = 1):
        """Record an agent call."""
        self.total_steps += attempts
        self.retries += max(0, attempts - 1)
        self.agents_called[agent] = self.agents_called.get(agent, 0) + 1
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "total_steps": self.total_steps,
            "max_depth_reached": self.max_depth_reached,
            "agents_called": self.agents_called,
            "retries": self.retries,
            "splits_created": self.splits_created,
            "branches_explored": self.branches_explored,
        }


# =============================================================================
# WORKING MEMORY - Core of the thinking-first architecture
# =============================================================================

@dataclass
class Hypothesis:
    """A claim the thinker believes, with supporting evidence."""
    claim: str
    confidence: float  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    source_iteration: int = 0


@dataclass
class WorkingMemory:
    """Persistent state that evolves across thinking iterations.
    
    GRAPH OF THOUGHTS ARCHITECTURE:
    ================================
    The Thinker operates like a DFS explorer with parallel splits:
    
    1. At any point, Thinker can decide to explore MULTIPLE parallel paths
       (e.g., search for 'vendeur' AND 'seller') up to max_branches
    
    2. Since we use rate-limited LLMs, parallelism is SIMULATED via DFS:
       - Create a split point recording all parallel branches
       - Dive deep into the first branch until done or max_depth
       - Backtrack to the split, explore next branch
       - Repeat until all branches explored
    
    3. After exploring all branches of a split, Thinker synthesizes findings
       and decides whether to continue or stop
    
    KEY LIMITS:
    - max_iterations: Number of Thinker decisions (not counting retries)
    - max_total_steps: Total agent calls including retries
    - max_depth: How deep any single branch can go before forced backtrack
    - max_branches: Max parallel branches at any split point
    
    NODE INDEXING:
    - Root level: 1, 2, 3, ... (max_branches)
    - Children of 1: 11, 12, 13, ...
    - Children of 12: 121, 122, 123, ...
    """
    
    # Graph of Thoughts structure
    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)  # index -> node
    current_node_index: str = ""   # Currently active node (empty = root)
    split_stack: List[SplitPoint] = field(default_factory=list)  # Stack of active splits
    
    # Statistics
    stats: ThinkingStats = field(default_factory=ThinkingStats)
    
    # Core reasoning state
    hypotheses: List[Hypothesis] = field(default_factory=list)
    patterns: Set[str] = field(default_factory=set)
    scripts_of_interest: List[Tuple[str, str]] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    dead_ends: List[Tuple[str, str]] = field(default_factory=list)
    
    # Execution trace
    iteration: int = 0            # Thinker decision count
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    raw_observations: List[Dict[str, Any]] = field(default_factory=list)
    
    # DFS exploration tracking
    exploration_stack: List[Dict[str, Any]] = field(default_factory=list)  # Stack of explored paths
    scripts_explored: Set[str] = field(default_factory=set)  # Scripts already visited
    current_depth: int = 0  # Current exploration depth
    last_node_id: str = ""  # ID of the last created node (for parent linking)
    
    # Limits (set from config)
    max_iterations: int = 5
    min_iterations: int = 1
    max_depth: int = 3
    max_branches: int = 4
    max_total_steps: int = 20
    confidence_threshold: float = 0.8
    
    # Outcome
    fail_reason: Optional[str] = None
    forced_stop_reason: Optional[str] = None  # Set when limits force a stop
    
    def get_current_depth(self) -> int:
        """Get the depth of the current exploration level."""
        return self.current_depth
    
    def get_next_child_index(self, parent_index: str = "") -> str:
        """Generate the next available child index for a parent.
        
        Examples:
        - Parent "" (root): returns "1", "2", "3", ...
        - Parent "1": returns "11", "12", "13", ...
        - Parent "12": returns "121", "122", "123", ...
        """
        if not parent_index:
            # Root level: count existing root children
            existing = [idx for idx in self.nodes.keys() if len(idx) == 1]
            next_num = len(existing) + 1
            return str(next_num)
        else:
            # Child level: count existing children of parent
            existing = [idx for idx in self.nodes.keys() 
                       if idx.startswith(parent_index) and len(idx) == len(parent_index) + 1]
            next_num = len(existing) + 1
            return f"{parent_index}{next_num}"
    
    def add_node(self, agent: str, question: str, description: str,
                 parent_index: str = None) -> ThoughtNode:
        """Create and add a new node to the graph."""
        if parent_index is None:
            parent_index = self.current_node_index
        
        index = self.get_next_child_index(parent_index)
        depth = len(index)
        
        node = ThoughtNode(
            index=index,
            parent_index=parent_index,
            agent=agent,
            description=description,
            question=question,
            depth=depth,
        )
        
        self.nodes[index] = node
        
        # Update parent's children list
        if parent_index and parent_index in self.nodes:
            self.nodes[parent_index].children.append(index)
        
        # Update stats
        self.stats.max_depth_reached = max(self.stats.max_depth_reached, depth)
        
        return node
    
    def create_split(self, branch_indices: List[str]) -> SplitPoint:
        """Create a new split point for parallel exploration."""
        split = SplitPoint(
            parent_index=self.current_node_index,
            branches=branch_indices.copy(),
            explored_branches=[],
            pending_branches=branch_indices.copy(),
            created_at_iteration=self.iteration,
        )
        self.split_stack.append(split)
        self.stats.splits_created += 1
        return split
    
    def backtrack_to_last_split(self) -> Optional[SplitPoint]:
        """Backtrack to the last split point with unexplored branches."""
        while self.split_stack:
            split = self.split_stack[-1]
            if split.pending_branches:
                return split
            # This split is exhausted, pop it
            self.split_stack.pop()
            self.stats.branches_explored += len(split.explored_branches)
        return None
    
    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Return the hypothesis with highest confidence."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)
    
    def can_continue(self) -> Tuple[bool, str]:
        """Check if we can continue exploring.
        
        Returns (can_continue, reason_if_not).
        """
        if self.stats.iterations >= self.max_iterations:
            return False, f"max_iterations ({self.max_iterations}) reached"
        if self.stats.total_steps >= self.max_total_steps:
            return False, f"max_total_steps ({self.max_total_steps}) reached"
        return True, ""
    
    def should_backtrack(self) -> bool:
        """Check if we should backtrack from current branch."""
        return self.get_current_depth() >= self.max_depth
    
    def should_stop(self) -> Tuple[bool, str]:
        """Determine if reasoning should terminate."""
        # Check forced limits
        can_go, reason = self.can_continue()
        if not can_go:
            self.forced_stop_reason = reason
            return True, reason
        
        # Must complete min_iterations
        if self.stats.iterations < self.min_iterations:
            return False, ""
        
        # Check confidence
        best = self.get_best_hypothesis()
        if best and best.confidence >= self.confidence_threshold:
            return True, f"confidence_threshold ({best.confidence:.2f})"
        
        # Check if all splits explored and no more questions
        if not self.split_stack and not self.open_questions and self.hypotheses:
            return True, "all_splits_explored"
        
        return False, ""
    
    def serialize(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for LLM prompts."""
        return {
            "iteration": self.iteration,
            "stats": self.stats.serialize(),
            "limits": {
                "max_iterations": self.max_iterations,
                "remaining_iterations": self.max_iterations - self.stats.iterations,
                "max_total_steps": self.max_total_steps,
                "remaining_steps": self.max_total_steps - self.stats.total_steps,
                "max_depth": self.max_depth,
                "current_depth": self.get_current_depth(),
                "max_branches": self.max_branches,
            },
            "current_node": self.current_node_index or "root",
            "active_splits": len(self.split_stack),
            "pending_branches": sum(len(s.pending_branches) for s in self.split_stack),
            "hypotheses": [
                {"claim": h.claim, "confidence": h.confidence, "evidence": h.evidence}
                for h in self.hypotheses
            ],
            "patterns": list(self.patterns),
            "scripts_of_interest": [
                {"script": s[0], "reason": s[1]} for s in self.scripts_of_interest
            ],
            "open_questions": self.open_questions,
            "nodes_count": len(self.nodes),
        }
    
    def render_tree(self) -> str:
        """Render the thought graph as a compact ASCII tree.
        
        Format: node_index (agent *N) where N is the number of attempts.
        """
        if not self.nodes:
            return "(empty)"
        
        lines = ["root"]
        
        def render_node(index: str, prefix: str, is_last: bool):
            node = self.nodes.get(index)
            if not node:
                return
            
            connector = "└─" if is_last else "├─"
            # Compact format: index (agent *N)
            attempts_str = f" *{node.attempts}" if node.attempts >= 1 else ""
            lines.append(f"{prefix}{connector} {index} ({node.agent}{attempts_str})")
            
            child_prefix = prefix + ("   " if is_last else "│  ")
            for i, child_idx in enumerate(node.children):
                render_node(child_idx, child_prefix, i == len(node.children) - 1)
        
        # Find root children (single digit indices)
        root_children = sorted([idx for idx in self.nodes.keys() if len(idx) == 1])
        for i, idx in enumerate(root_children):
            render_node(idx, "", i == len(root_children) - 1)
        
        return "\n".join(lines)


# =============================================================================
# THINKER PROMPT - Single LLM that reasons about full state
# =============================================================================

DEFAULT_THINKER_PROMPT = (
    COMMON_ORACLE_PRIMER
    + """
You are the "Thinker" - the central reasoning engine of Oracle. You see the FULL working memory
state and make intelligent decisions using a GRAPH OF THOUGHTS with parallel splits.

## GRAPH OF THOUGHTS ARCHITECTURE
You explore like a DFS with parallel branches (SPLITS):

### Node Indexing
- Root children: 1, 2, 3, ... (up to max_branches = {max_branches})
- Children of node 1: 11, 12, 13, ...
- Children of node 12: 121, 122, 123, ...

### Exploration Strategy
1. **DISCOVER** (depth 0→1): Find candidate scripts via semantic/syntax search
2. **SPLIT** (optional): Create parallel branches to explore alternatives simultaneously
   - Example: Search for BOTH "vendeur" AND "seller" in parallel
   - Each split can have up to {max_branches} branches
3. **DIVE** (depth +1): Go deeper into ONE branch
4. **BACKTRACK** (depth -1): When branch is done OR max_depth reached, go back to last split
5. **SYNTHESIZE**: After all branches explored, combine findings

### LIMITS YOU MUST RESPECT
- **Iterations remaining**: {remaining_iterations} of {max_iterations}
  → One iteration = one Thinker decision (retries don't count as iterations)
- **Steps remaining**: {remaining_steps} of {max_total_steps}
  → One step = one agent call (including retries)
- **Current depth**: {current_depth} of max {max_depth}
  → When max_depth reached, you MUST backtrack (don't stop!)
- **Max branches per split**: {max_branches}

### WHEN FORCED TO STOP BY LIMITS
If you must stop due to limits, you MUST still provide a synthesis:
- Summarize what you found so far
- Explain that you couldn't fully explore due to [iteration/step/depth] limits
- Give your best answer with reduced confidence

Current exploration:
- Node: {current_node}
- Depth: {current_depth}/{max_depth}
- Active splits: {active_splits}
- Pending branches: {pending_branches}

## Your Role
You operate in a think → act → observe → update loop:
1. THINK: Analyze current state, hypotheses, patterns, and evidence
2. ACT: Decide ONE action (semantic, syntax, read, dive, or backtrack)
3. OBSERVE: (The system executes and returns results)
4. UPDATE: Revise hypotheses based on new evidence

## CRITICAL CONTEXT: Codebase Language
The Envision codebase uses ENGLISH for code, comments, variable names, and show table titles.
User questions are often in FRENCH. You MUST translate concepts:
- "meilleurs vendeurs" → "Best Sellers", "TopSellers", "BestSeller"
- "analyse des ventes" → "Sales Analysis", "Sales Overview"
- Always search for BOTH the French term AND its English equivalent

## Available Actions

### 1. SEMANTIC - Discover scripts by concept (best for depth 0)
Use when you don't know WHERE to look - discovers relevant scripts by meaning.
Good for: "Find scripts about Best Sellers or top selling items"

### 2. SYNTAX - Search for specific patterns
Use when you have an EXACT pattern to find across all scripts.
Good for: "Search for BestSeller or TopSeller in all .nvn files"

### 3. READ - Read a specific script's content (increases depth)
Use when you've found an interesting script and want to understand its logic.
Provide the script path or ID. This is a DIVE action - it increases depth.
Good for: "Read script 67982 to understand how BestSellerRank is calculated"

### 4. DIVE - Go deeper into a specific script (increases depth)
Use after discovering a script to focus your search WITHIN that script only.
Subsequent actions will be scoped to this script until you backtrack.
Good for: "Dive into script 67982 to explore its Best Seller calculations"

### 5. BACKTRACK - Return to parent exploration level (decreases depth)
Use when you've finished exploring the current branch and want to try other paths.
Good for: "Backtrack to explore other candidate scripts"

### 6. SPLIT - Create parallel branches to explore alternatives
Use when you want to explore MULTIPLE paths in parallel (e.g., French AND English terms).
Provide a list of branch descriptions. System will simulate parallelism via DFS.
Good for: "Split to search 'vendeur' in one branch and 'seller' in another"

### 7. COMPUTE - Delegate computation to a safe code executor
Use when you need to perform any calculation, counting, or data transformation.
The code executor can run simple Python to process data reliably.
ALWAYS USE THIS when:
- Counting unique/distinct values from a list
- Extracting unique items from collected data
- Any mathematical computation
- Processing lists (filtering, sorting, deduplicating)
- Computing statistics (sum, average, min, max)
Provide the data to process and describe the computation needed.
Good for: "Count unique modules from this list: [...]"
Good for: "Extract distinct variable names from these 56 imports"

## Working Memory State
{working_memory}

## Original Question
{question}

## Preprocessed Context
- Normalized: {normalized_question}
- Clarifier: {clarifier}
- Decomposition: {decomposition}
- Router recommendation: {router}

## Last Observation (if any)
{last_observation}

## Your Task
Analyze the current state and decide ONE of:
1. **Execute an action** to gather more evidence
2. **Create a SPLIT** to explore multiple paths in parallel
3. **Stop** if you have sufficient confidence in an answer

Return strictly valid JSON:
{{
    "thinking": "Your internal reasoning (50-100 words). Consider: What depth are we at? Should we dive, split, or backtrack? What's missing? How many iterations/steps do we have left?",
    "decision": "act" | "split" | "stop" | "fail",
    "stop_reason": "Only if decision=stop: why are we confident enough to answer?",
    "fail_reason": "Only if decision=fail: why we cannot provide a reliable answer (agent issues, low confidence, contradictory evidence...)",
    "forced_stop_synthesis": "Only if limits force you to stop: summarize findings and explain what you couldn't explore",
    
    "action": {{
        "type": "semantic" | "syntax" | "read" | "dive" | "backtrack" | "retry" | "split" | "compute",
        "target_script": "For read/dive: script path or ID to focus on",
        "question": "NATURAL LANGUAGE QUESTION for the agent",
        "data": "For compute: the data to process (e.g., list of items, numbers)",
        "context": "Optional context to guide the agent",
        "hints": ["English terms to search", "BestSeller", "TopSeller"],
        "rationale": "Why this specific action will help. If diving: what do you expect to find?",
        "correction_feedback": "Only for retry: what was wrong with the previous attempt and how to fix it",
        "branches": [
            {{"description": "Search for French term 'vendeur'", "action_type": "syntax", "query": "vendeur"}},
            {{"description": "Search for English term 'seller'", "action_type": "syntax", "query": "seller"}}
        ]
    }},
    
    "result_critique": {{
        "command_was_appropriate": true | false,
        "issue": "If false: describe what was wrong (e.g., 'used keyword=write but question asked for all references')",
        "should_retry": true | false,
        "reformulated_question": "If should_retry: improved question for the agent"
    }},
    
    "memory_updates": {{
        "new_hypotheses": [
            {{"claim": "What you now believe", "confidence": 0.0-1.0, "evidence": ["supporting facts"]}}
        ],
        "updated_confidences": [
            {{"claim": "existing claim text", "new_confidence": 0.0-1.0, "reason": "why changed"}}
        ],
        "new_patterns": ["newly discovered patterns like 'Best Sellers', column names"],
        "new_scripts_of_interest": [{{"script": "path or id", "reason": "why interesting"}}],
        "resolved_questions": ["questions we can now answer"],
        "new_questions": ["new sub-questions discovered"],
        "new_dead_ends": [{{"path": "abandoned path", "reason": "why abandoned"}}]
    }}
}}

## EXPLORATION GUIDELINES
- **DEPTH 0**: Start with semantic or syntax to discover candidate scripts
- **DEPTH 1+**: Dive into promising scripts to understand their logic
- **USE SPLIT** when you need to explore multiple alternatives (e.g., synonyms, translations)
- **BEFORE STOPPING**: Ensure you've explored multiple branches if confidence < 0.9
- **USE READ/DIVE** to understand HOW a calculation works, not just WHERE it is
- **BACKTRACK** when:
  - You want to go back to a parent node to create a SPLIT for alternatives
  - Max depth ({max_depth}) is reached → MANDATORY backtrack
  - You want to explore the next branch of an active split
  - **NEVER backtrack just because last action failed - use RETRY first!**
- **CONFIDENCE GUIDE**:
  - 0.5-0.6: Found relevant scripts but haven't explored them deeply
  - 0.7-0.8: Explored 1-2 scripts in depth, found key patterns
  - 0.9+: Explored multiple branches, cross-verified, understand the logic

## CRITICAL: RETRY vs BACKTRACK LOGIC
**When an agent action fails or returns empty results:**

1. **FIRST: Analyze the action** - Was the approach correct?
   - Check result_critique.command_was_appropriate
   - If the code/command was GOOD but data was empty → RETRY (system may resolve data better)
   - If the code/command was WRONG → RETRY with correction_feedback

2. **RETRY on same node** (attempts increment):
   - Use action.type = "retry" with correction_feedback
   - The system tracks attempts per node
   - You can retry up to {max_agent_retries} times per action type

3. **BACKTRACK only when**:
   - You've exhausted retries (max_agent_retries reached)
   - You want to CREATE A SPLIT to explore alternative approaches
   - Backtracking goes to the PARENT node, not to re-execute the parent
   - From parent, you can create a SPLIT to explore other paths

4. **Parent nodes are VALIDATED**:
   - If you went from syntax → compute, it means you VALIDATED syntax results
   - Backtracking to syntax does NOT re-execute syntax
   - It positions you to create a SPLIT for alternative approaches

**Example flow for "Quels modules?"**:
- syntax: returns 56 occurrences ✓ (validated by going to compute)
- compute: returns [] (empty - data resolution issue)
- → RETRY compute (not backtrack!) with better data reference
- → If compute still fails after {max_agent_retries} retries
- → BACKTRACK to syntax parent, then SPLIT to try different approach

## BACKTRACKING AND SPLITS
When you hit max_depth ({max_depth}), you MUST backtrack - don't stop!
After backtracking from a split:
- The system will present you with the next pending branch to explore
- Explore it until done, then backtrack again
- When all branches of a split are explored, synthesize and decide next step

## CRITICAL: NEVER STOP WITH EMPTY RESULTS
Before deciding to STOP, you MUST check the last_observation:
- If `scripts_count: 0` or `occurrences: 0` → DO NOT STOP, the search failed
- If `results: []` or `computed_result: []` → DO NOT STOP, no evidence was found
- If `success: false` → DO NOT STOP, the action failed
- If results contradict your hypothesis → DO NOT STOP with high confidence

**You can only STOP if you have ACTUAL EVIDENCE that supports your answer.**
An empty result is NOT evidence - it means you need to RETRY or try a different approach!

## CRITICAL: RESULT VALIDATION
After each agent action, you MUST analyze BOTH the results AND the commands used:

1. **Command Appropriateness**: Did the agent use the right approach?
   - For "font mention de" / "mentionnent" → command should NOT have keyword=read or keyword=write
   - For "lisent" / "read" → keyword=read is correct
   - For counting questions → need complete list, not filtered

2. **Result Completeness**: Are the results likely complete?
   - If command was too restrictive (e.g., keyword=write when asking for all mentions), results are INCOMPLETE
   - If grep was used for path search instead of scanner, results may be INCOMPLETE

3. **When to RETRY**:
   - Agent used wrong parameters (e.g., keyword when not appropriate)
   - Agent missed the point of the question
   - Results seem suspiciously low (e.g., 1 script when question implies multiple)
   - **Results are EMPTY (scripts_count: 0)** → MUST retry with different approach
   - Use "retry" action type with correction_feedback explaining the issue

4. **When to FAIL** (decision="fail"):
   - Multiple retries failed to get reliable results
   - Evidence is contradictory and cannot be resolved
   - Cannot achieve sufficient confidence despite exploration
   - Agent consistently misinterprets the question
   - ALWAYS provide fail_reason explaining why answer cannot be guaranteed

## MANDATORY: COMPUTE AGENT FOR ALL DATA OPERATIONS
**YOU ARE FORBIDDEN from counting, deduplicating, or extracting values yourself!**
You are an LLM - you WILL hallucinate counts. The compute agent runs REAL Python code.

**MANDATORY compute BEFORE stop when:**
- Question asks "quels" / "which" / "what" items → compute extracts the list
- Question asks "combien" / "how many" → compute counts
- Question asks for "unique" / "distinct" values → compute deduplicates
- You have raw data (occurrences_data) and need to answer → compute processes it
- ANY list extraction or counting task → compute handles it

**YOU CANNOT STOP without compute if:**
- The question requires listing, counting, or extracting data
- You have occurrences_data from syntax but haven't processed it with compute
- You're about to claim "X modules found" without compute validation

**Syntax agent returns RAW DATA. Compute agent PROCESSES it.**
Syntax → raw occurrences_data (may have duplicates, needs extraction)
Compute → clean results (deduplicated, counted, extracted)

**Example workflow for "Quels modules sont utilisés?":**
1. syntax: scan imports → returns occurrences_data with resolved_path fields
2. compute: "Extract unique values from resolved_path field" → returns clean list
3. stop: with the compute-validated result

**In the compute action, provide:**
- question: What computation to perform (e.g., "Extract unique resolved_path values from occurrences_data")
- data: Reference the data from last_observation (the system will pass it)

## Pattern Hints for Common Concepts
| French | English Code Patterns |
|--------|----------------------|
| meilleurs vendeurs | BestSeller, TopSeller, Best Sellers, Rank |
| analyse magasins | Stores Analysis, Location, Store |
| analyse ventes | Sales Analysis, Orders, NetAmount |
| catalogue | Catalog, Items, Products |
| inspecteur | Inspector, Dashboard, Analysis |
"""
).strip()


@dataclass
class RunOutcome:
    """Container bundling the final payload plus telemetry for one run."""

    payload: Dict[str, Any]
    text: str
    decision: ScopeDecision
    preprocess: Optional[PreprocessBundle]
    thought_report: Optional[Dict[str, Any]]
    stats: Optional[Dict[str, Any]]


class MainAgent:
    """High-level orchestrator that chains preprocess + thought-graph reasoning."""

    def __init__(self, config: Dict[str, Any], *, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        main_cfg = config.get("main_agent", {})
        scope_cfg = main_cfg.get("scope", {})
        self.llm_defaults = config.get("llm", {})
        paths_cfg = config.get("paths", {})
        self.script_root = Path(paths_cfg.get("script_mirror", "env_scripts")).resolve()
        self.mapping_path = Path(paths_cfg.get("mapping_file", "mapping.txt")).resolve()
        self.identity = main_cfg.get("identity", "Oracle")
        self.max_agent_retries = max(1, int(main_cfg.get("max_agent_retries", 2)))
        self.llm_registry: Dict[str, str] = {}
        self.project_scope = scope_cfg.get(
            "project_scope",
            "Analysis of Envision (.nvn) scripts, deterministic tooling (grep, FAISS, index build), and /Clean/... assets.",
        )
        self.greeting_reply = scope_cfg.get(
            "greeting_reply",
            "Hello, I am Oracle, guardian of the Envision scripts. How can I help today?",
        )
        self.off_topic_reply = scope_cfg.get(
            "off_topic_reply",
            "I only handle Envision scripts and deterministic investigations for the LLM-DSL workspace.",
        )
        self.scope_prompt_template = scope_cfg.get("prompt_template", DEFAULT_SCOPE_PROMPT)
        logging_cfg = main_cfg.get("logging", {})
        history_hint = logging_cfg.get("history_file", "logs/oracle-history.jsonl")
        self.history_file = Path(history_hint).resolve() if history_hint else None
        if self.history_file:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.scope_agent = self._prepare_component_llm("scope", scope_cfg)

        qp_cfg = config.get("question_preprocess", {})
        clarifier_cfg = qp_cfg.get("clarifier", {})
        decomposer_cfg = qp_cfg.get("decomposer", {})
        router_cfg = qp_cfg.get("router", {})

        self.clarifier: Optional[LLMQuestionClarifier] = None
        if clarifier_cfg.get("enabled", True):
            clarifier_agent = self._prepare_component_llm("clarifier", clarifier_cfg)
            self.clarifier = LLMQuestionClarifier(llm=clarifier_agent)

        self.decomposer: Optional[LLMQuestionDecomposer] = None
        if decomposer_cfg.get("enabled", True):
            decomposer_agent = self._prepare_component_llm("decomposer", decomposer_cfg)
            self.decomposer = LLMQuestionDecomposer(llm=decomposer_agent)

        self.router: Optional[LLMRouter] = None
        if router_cfg.get("enabled", True):
            router_agent = self._prepare_component_llm("router", router_cfg)
            self.router = LLMRouter(llm=router_agent)

        workers_cfg = main_cfg.get("workers", {})
        semantic_cfg = workers_cfg.get("semantic", {})
        syntax_cfg = workers_cfg.get("syntax", {})

        self.semantic_agent: Optional[SemanticAgent] = None
        self.semantic_default_top_k = int(semantic_cfg.get("max_results", 5))
        if semantic_cfg.get("enabled", True):
            try:
                self.semantic_agent = SemanticAgent(default_top_k=self.semantic_default_top_k)
            except Exception as exc:  # pragma: no cover - runtime env issues
                if self.verbose:
                    print(f"[warn] Semantic agent unavailable: {exc}")

        self.syntax_agent: Optional[SyntaxAgent] = None
        self.syntax_max_commands = int(syntax_cfg.get("max_commands", 3))
        if syntax_cfg.get("enabled", True):
            try:
                syntax_llm = self._prepare_component_llm("syntax_worker", syntax_cfg)
                self.syntax_agent = SyntaxAgent(
                    llm=syntax_llm,
                    script_root=self.script_root,
                    max_commands=self.syntax_max_commands,
                    mapping_file=self.mapping_path,
                )
            except Exception as exc:  # pragma: no cover - runtime env issues
                if self.verbose:
                    print(f"[warn] Syntax agent unavailable: {exc}")

        allowed_actions: List[str] = []
        if self.semantic_agent:
            allowed_actions.append("semantic")
        if self.syntax_agent:
            allowed_actions.append("syntax")
        
        # Code agent - always enabled for computation tasks
        code_cfg = workers_cfg.get("code", {})
        self.code_agent: Optional[CodeAgent] = None
        if code_cfg.get("enabled", True):
            try:
                code_agent_llm = self._prepare_component_llm("code", code_cfg)
                self.code_agent = CodeAgent(llm=code_agent_llm)
                allowed_actions.append("compute")
            except Exception as exc:  # pragma: no cover - runtime env issues
                if self.verbose:
                    print(f"[warn] Code agent unavailable: {exc}")
        
        self.available_actions = allowed_actions

        thought_cfg = main_cfg.get("thought_graph", {})
        self.thought_enabled = thought_cfg.get("enabled", True)
        self.thought_prompt_template = thought_cfg.get("prompt_template", DEFAULT_THOUGHT_PROMPT)
        max_iterations = int(thought_cfg.get("max_iterations", 3))
        min_iterations = int(thought_cfg.get("min_iterations", 1))
        if min_iterations < 1:
            min_iterations = 1
        if min_iterations > max_iterations:
            min_iterations = max_iterations
        self.thought_limits = ThoughtLimits(
            max_iterations=max_iterations,
            min_iterations=min_iterations,
            max_branches=int(thought_cfg.get("max_branches", 2)),
            max_depth=int(thought_cfg.get("max_depth", 3)),
            max_total_steps=int(thought_cfg.get("max_total_steps", 6)),
        )
        self.thought_agent = (
            self._prepare_component_llm("thought_graph", thought_cfg)
            if self.thought_enabled
            else None
        )
        # Enable the new thinking-first architecture by default
        # Set to False to use legacy thought_graph parallel expansion
        self.use_thinking_loop = thought_cfg.get("use_thinking_loop", True)
        
        self.script_mapping = load_mapping(self.mapping_path)
        self.path_to_id = self._build_path_index(self.script_mapping)

        answer_cfg = main_cfg.get("synthesizer", {}) if isinstance(main_cfg, dict) else {}
        self.synthesizer_inline_limit = max(1, int(answer_cfg.get("max_inline_refs", 5)))
        self.synthesizer_prompt_template = answer_cfg.get("prompt_template", DEFAULT_SYNTHESIZER_PROMPT)
        self.synthesizer: Optional[LLMClient] = None
        if answer_cfg.get("enabled", True):
            try:
                self.synthesizer = self._prepare_component_llm("synthesizer", answer_cfg)
            except Exception as exc:  # pragma: no cover - optional component
                if self.verbose:
                    print(f"[warn] Synthesizer disabled: {exc}")

        # Verbose output limits (lines to display for each agent type)
        verbose_cfg = main_cfg.get("verbose", {})
        self.verbose_limits = {
            "syntax": int(verbose_cfg.get("syntax_max_lines", 40)),
            "compute": int(verbose_cfg.get("compute_max_lines", 30)),
            "semantic": int(verbose_cfg.get("semantic_max_lines", 20)),
            "default": int(verbose_cfg.get("default_max_lines", 20)),
        }

        if self.verbose:
            self._print_llm_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_question(self, question: str, *, context: str = "") -> str:
        outcome = self.process_question(
            question,
            context=context,
            as_text=True,
            record_history=True,
        )
        return outcome.text

    def run_question(
        self,
        question: str,
        *,
        context: str = "",
        record_history: bool = False,
    ) -> RunOutcome:
        """Execute the pipeline and expose the structured payload plus telemetry."""
        return self.process_question(
            question,
            context=context,
            as_text=False,
            record_history=record_history,
        )

    def process_question(
        self,
        question: str,
        *,
        context: str = "",
        as_text: bool,
        record_history: bool,
    ) -> RunOutcome:
        """Process one question end-to-end through scope -> preprocess -> thought graph.

        The method orchestrates every data structure defined in this module and can optionally
        skip history logging when reused by external scripts (e.g., benchmarks).
        """
        context = context or ""
        stripped = question.strip()
        if not stripped:
            payload = self._build_simple_response(
                "empty_question",
                "The question is empty. Please rephrase.",
            )
            decision = ScopeDecision(status="invalid", reply="", reason="empty_question")
            return self._emit_response(
                payload=payload,
                decision=decision,
                question=stripped,
                context=context,
                preprocess=None,
                thought_report=None,
                loggable=False,
                as_text=as_text,
                record_history=record_history,
            )

        decision = self._classify_question(stripped, context)
        decision_dict = asdict(decision)
        if self.verbose:
            self._print_scope_decision(decision)

        if decision.status == "greeting":
            payload = self._build_simple_response(
                "greeting",
                decision.reply or self.greeting_reply,
                {"scope_decision": decision_dict},
            )
            return self._emit_response(
                payload=payload,
                decision=decision,
                question=stripped,
                context=context,
                preprocess=None,
                thought_report=None,
                loggable=False,
                as_text=as_text,
                record_history=record_history,
            )
        if decision.status == "off_topic":
            payload = self._build_simple_response(
                "off_topic",
                decision.reply or self.off_topic_reply,
                {"scope_decision": decision_dict},
            )
            return self._emit_response(
                payload=payload,
                decision=decision,
                question=stripped,
                context=context,
                preprocess=None,
                thought_report=None,
                loggable=False,
                as_text=as_text,
                record_history=record_history,
            )
        if decision.status != "project":
            payload = self._build_simple_response(
                "unknown_scope",
                decision.reply or "I cannot process this request right now.",
                {"scope_decision": decision_dict},
            )
            return self._emit_response(
                payload=payload,
                decision=decision,
                question=stripped,
                context=context,
                preprocess=None,
                thought_report=None,
                loggable=False,
                as_text=as_text,
                record_history=record_history,
            )

        preprocess_bundle = self._run_question_preprocess(stripped, context)
        if self.verbose:
            self._print_preprocess_stages(preprocess_bundle.stages)

        if not preprocess_bundle.router:
            payload = self._build_simple_response(
                "router_error",
                "Router could not determine downstream actions.",
                {
                    "scope_decision": decision_dict,
                    "preprocess": preprocess_bundle.serialize(),
                },
            )
            return self._emit_response(
                payload=payload,
                decision=decision,
                question=stripped,
                context=context,
                preprocess=preprocess_bundle,
                thought_report=None,
                loggable=False,
                as_text=as_text,
                record_history=record_history,
            )

        # Choose between thinking-first loop (new) or thought graph (legacy)
        # The thinking-first loop uses WorkingMemory and iterative refinement
        use_thinking_loop = getattr(self, 'use_thinking_loop', True)
        
        if use_thinking_loop:
            memory, thought_report = self._run_thinking_loop(stripped, context, preprocess_bundle)
            # Enrich report with memory-derived data for downstream synthesis
            thought_report["working_memory"] = memory.serialize()
        else:
            thought_report = self._run_thought_graph(stripped, context, preprocess_bundle)
            memory = None
        
        structured_payload, references_payload, agent_stats = self._build_structured_response(
            question=stripped,
            decision=decision,
            preprocess=preprocess_bundle,
            thought_report=thought_report,
            context=context,
            working_memory=memory,
        )
        if self.verbose:
            self._print_thought_graph_report(thought_report)
            self._print_thought_tree(thought_report)
            self._print_final_result_stats(
                stripped, decision, preprocess_bundle, thought_report, structured_payload
            )
        log_stats = self._build_log_stats(
            agent_stats=agent_stats,
            references=references_payload,
            payload=structured_payload,
        )
        return self._emit_response(
            payload=structured_payload,
            decision=decision,
            question=stripped,
            context=context,
            preprocess=preprocess_bundle,
            thought_report=thought_report,
            stats=log_stats,
            loggable=structured_payload.get("status") == "completed",
            as_text=as_text,
            record_history=record_history,
        )

    def repl(self) -> None:
        """Run an interactive loop that repeatedly calls :meth:`handle_question`."""
        print("Oracle interactive session. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("dsl> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession closed.")
                break
            if not user_input:
                continue
            if user_input in {"exit", "quit"}:
                print("Goodbye!")
                break
            response = self.handle_question(user_input)
            print(f"\n>>> {response}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _classify_question(self, question: str, context: str) -> ScopeDecision:
        """Call the scope LLM to decide if the request is greeting, project, or off-topic."""
        if not self.scope_agent:
            return ScopeDecision(
                status="off_topic",
                reply=self.off_topic_reply,
                reason="Scope agent unavailable",
            )
        prompt = self.scope_prompt_template.format(
            identity=self.identity,
            question=question,
            context=context or "(no additional context)",
            project_scope=self.project_scope,
        )
        try:
            raw = self.scope_agent.generate_response(prompt=prompt)
            parsed = parse_llm_json(raw)
        except Exception as exc:  # pragma: no cover - defensive clause
            return ScopeDecision(
                status="off_topic",
                reply=self.off_topic_reply,
                reason=f"Unable to contact Oracle ({exc})",
            )
        status = (parsed.get("status") or "project").strip().lower()
        reply = parsed.get("reply") or (
            self.greeting_reply if status == "greeting" else self.off_topic_reply
        )
        reason = parsed.get("reason") or "Oracle classification"
        if status not in {"greeting", "project", "off_topic"}:
            status = "project"
        return ScopeDecision(status=status, reply=reply, reason=reason)

    def _run_question_preprocess(self, question: str, context: str) -> PreprocessBundle:
        """Execute clarifier -> decomposer -> router chain and collect a :class:`PreprocessBundle`."""
        bundle = PreprocessBundle(normalized_question=question)
        stages: List[StageExecution] = []
        normalized_question = question

        if self.clarifier:
            payload, success = self._run_stage_with_retries(
                name="clarifier",
                stages=stages,
                runner=lambda: asdict(
                    self.clarifier.assess(
                        QuestionClarifierInput(question=question, context=context or None)
                    )
                ),
                justification="Clarity guard output",
            )
            if not success:
                bundle.stages = stages
                return bundle
            normalized_question = (
                (payload or {}).get("normalized_question", "").strip() or normalized_question
            )
            bundle.clarifier = payload

        if self.decomposer:
            payload, success = self._run_stage_with_retries(
                name="decomposer",
                stages=stages,
                runner=lambda: asdict(
                    self.decomposer.decompose(
                        QuestionDecomposerInput(
                            question=normalized_question,
                            context=context or None,
                        )
                    )
                ),
                justification="Question decomposition",
            )
            if not success:
                bundle.stages = stages
                bundle.normalized_question = normalized_question
                return bundle
            bundle.decomposition = payload

        if self.router:
            payload, success = self._run_stage_with_retries(
                name="router",
                stages=stages,
                runner=lambda: asdict(
                    self.router.route(
                        RouterInput(question=normalized_question, context=context or None)
                    )
                ),
                justification="Initial routing suggestion",
            )
            if success:
                bundle.router = payload

        bundle.stages = stages
        bundle.normalized_question = normalized_question
        return bundle

    def _run_stage_with_retries(
        self,
        *,
        name: str,
        stages: List[StageExecution],
        runner: Callable[[], Dict[str, Any]],
        justification: str,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Run a preprocess stage with retry logic and append the resulting :class:`StageExecution`."""
        attempts = 0
        last_error = ""
        while attempts < self.max_agent_retries:
            attempts += 1
            try:
                payload = runner() or {}
                stages.append(
                    StageExecution(
                        name=name,
                        success=True,
                        payload=payload,
                        justification=justification,
                        attempts=attempts,
                    )
                )
                return payload, True
            except LLMCommunicationError as exc:
                last_error = str(exc)
        stages.append(
            StageExecution(
                name=name,
                success=False,
                payload=None,
                justification=last_error or f"{name} failed",
                attempts=max(attempts, 1),
            )
        )
        return None, False

    def _run_thought_graph(
        self,
        question: str,
        context: str,
        bundle: PreprocessBundle,
    ) -> Dict[str, Any]:
        """Drive the iterative planner loop and return the serialized planner report."""
        if not self.available_actions:
            return {
                "status": "skipped",
                "reason": "No downstream semantic/syntax agents are available.",
                "steps": [],
                "pending_router_actions": [],
            }

        router_payload = bundle.router or {}
        router_actions_raw = router_payload.get("next_actions", [])
        pending_router_actions = [
            action
            for action in (
                ROUTER_ACTION_MAP.get(action_name)
                for action_name in router_actions_raw
            )
            if action in self.available_actions
        ]
        if not pending_router_actions:
            pending_router_actions = list(self.available_actions)

        if not self.thought_enabled or not self.thought_agent:
            steps = self._fallback_router_execution(bundle, pending_router_actions)
            return {
                "status": "fallback",
                "reason": "Thought planner disabled or unavailable.",
                "steps": [step.serialize() for step in steps],
                "pending_router_actions": [action for action in pending_router_actions if action],
            }

        used_ids: Set[str] = set()
        history: List[Dict[str, Any]] = []
        node_depths: Dict[str, int] = {"root": 0}
        display_ids: Dict[str, str] = {"root": ""}
        child_counters: Dict[str, int] = {}
        executed_steps: List[ThoughtStepResult] = []
        iteration_count = 0
        final_decision = "stop"
        final_reason = "Max iterations reached."

        while iteration_count < self.thought_limits.max_iterations:
            # Ask the planner LLM how to expand the current thought graph frontier.
            plan = self._request_thought_plan(
                question=question,
                context=context,
                bundle=bundle,
                history=history,
                used_ids=used_ids,
                pending_router_actions=pending_router_actions,
            )
            iteration_count += 1
            final_decision = plan.decision
            final_reason = plan.reason

            if not plan.steps:
                if plan.decision != "continue":
                    if iteration_count < self.thought_limits.min_iterations:
                        final_reason = (
                            "Planner stopped before reaching minimum iterations."
                        )
                    break
                final_decision = "stop"
                final_reason = "Planner returned no nodes."
                break

            for step in plan.steps:
                parent_id = step.parent_id or "root"
                parent_depth = node_depths.get(parent_id, 0)
                depth = parent_depth + 1
                if depth > self.thought_limits.max_depth:
                    continue
                used_ids.add(step.node_id)
                node_depths[step.node_id] = depth
                display_id = self._assign_display_id(
                    node_id=step.node_id,
                    parent_id=parent_id,
                    display_ids=display_ids,
                    child_counters=child_counters,
                )
                result = self._execute_thought_step(
                    step, bundle, depth, display_id
                )
                executed_steps.append(result)
                # ``history`` mirrors the serialized ``ThoughtStepResult`` list that the
                # planner will receive on the next iteration.
                history.append(
                    {
                        "id": result.node_id,
                        "parent": result.parent_id,
                        "action": result.action,
                        "depth": result.depth,
                        "success": result.success,
                        "summary": result.summary,
                        "justification": result.justification,
                        "display_id": result.display_id,
                        "attempts": result.attempts,
                    }
                )
                if step.action in pending_router_actions:
                    pending_router_actions.remove(step.action)
                if len(executed_steps) >= self.thought_limits.max_total_steps:
                    final_decision = "stop"
                    final_reason = "Reached maximum allowed thought steps."
                    break

            if len(executed_steps) >= self.thought_limits.max_total_steps:
                break

            if iteration_count < self.thought_limits.min_iterations:
                continue

            if final_decision == "stop":
                break

        if len(executed_steps) < self.thought_limits.min_iterations:
            needed = self.thought_limits.min_iterations - len(executed_steps)
            self._enforce_minimum_steps(
                executed_steps=executed_steps,
                needed=needed,
                bundle=bundle,
                display_ids=display_ids,
                child_counters=child_counters,
                used_ids=used_ids,
                history=history,
                pending_router_actions=pending_router_actions,
            )

        return {
            "status": "completed",
            "iterations": iteration_count,
            "decision": final_decision,
            "reason": final_reason,
            "pending_router_actions": pending_router_actions,
            "steps": [step.serialize() for step in executed_steps],
        }

    def _assign_display_id(
        self,
        *,
        node_id: str,
        parent_id: str,
        display_ids: Dict[str, str],
        child_counters: Dict[str, int],
    ) -> str:
        """Produce a human-readable tree index (1, 1.1, 1.2, ...) for a planner node."""
        parent_display = display_ids.get(parent_id, "")
        key = parent_display
        next_index = child_counters.get(key, 0) + 1
        child_counters[key] = next_index
        display_id = f"{parent_display}{next_index}" if parent_display else str(next_index)
        display_ids[node_id] = display_id
        return display_id

    def _fallback_router_execution(
        self, bundle: PreprocessBundle, actions: List[str]
    ) -> List[ThoughtStepResult]:
        """Execute router-suggested actions sequentially when the planner is unavailable."""
        results: List[ThoughtStepResult] = []
        display_ids: Dict[str, str] = {"root": ""}
        child_counters: Dict[str, int] = {}
        unique_actions = actions or self.available_actions
        for idx, action in enumerate(unique_actions, start=1):
            node_id = f"fallback-{idx}"
            step = ThoughtStepRequest(
                node_id=node_id,
                parent_id="root",
                action=action,
                question=bundle.normalized_question,
                context=None,
                hints=[],
                max_results=None,
                justification="Router default execution",
            )
            display_id = self._assign_display_id(
                node_id=node_id,
                parent_id="root",
                display_ids=display_ids,
                child_counters=child_counters,
            )
            result = self._execute_thought_step(
                step, bundle, depth=1, display_id=display_id
            )
            results.append(result)
        return results

    def _request_thought_plan(
        self,
        *,
        question: str,
        context: str,
        bundle: PreprocessBundle,
        history: List[Dict[str, Any]],
        used_ids: Set[str],
        pending_router_actions: List[str],
    ) -> ThoughtPlan:
        """Prompt the planner LLM and convert its JSON into a :class:`ThoughtPlan`."""
        if not self.thought_agent:
            return ThoughtPlan(decision="stop", reason="Planner unavailable", steps=[])

        prompt = self.thought_prompt_template.format(
            max_branches=self.thought_limits.max_branches,
            max_depth=self.thought_limits.max_depth,
            max_total_steps=self.thought_limits.max_total_steps,
            allowed_actions=", ".join(self.available_actions) or "(none)",
            router_actions=json.dumps(pending_router_actions, ensure_ascii=False),
            used_ids=json.dumps(sorted(used_ids), ensure_ascii=False),
            question=question,
            normalized_question=bundle.normalized_question,
            context=context or "(no additional context)",
            clarifier=json.dumps(bundle.clarifier or {}, ensure_ascii=False),
            decomposition=json.dumps(bundle.decomposition or {}, ensure_ascii=False),
            router=json.dumps(bundle.router or {}, ensure_ascii=False),
            history=json.dumps(history, ensure_ascii=False),
        )
        try:
            raw = self.thought_agent.generate_response(prompt=prompt)
            parsed = parse_llm_json(raw)
        except Exception as exc:  # pragma: no cover - planner guard
            return ThoughtPlan(
                decision="stop",
                reason=f"Thought planner failed: {exc}",
                steps=[],
            )

        decision = str(parsed.get("decision", "stop")).strip().lower()
        reason = str(parsed.get("reason", "Planner response missing reason.")).strip()
        raw_nodes = parsed.get("nodes") or []
        steps = self._parse_thought_nodes(raw_nodes, used_ids)
        return ThoughtPlan(decision=decision, reason=reason, steps=steps)

    def _parse_thought_nodes(
        self,
        raw_nodes: List[Dict[str, Any]],
        used_ids: Set[str],
    ) -> List[ThoughtStepRequest]:
        """Validate planner nodes and turn them into :class:`ThoughtStepRequest` objects."""
        steps: List[ThoughtStepRequest] = []
        if not isinstance(raw_nodes, list):
            return steps
        for entry in raw_nodes:
            if len(steps) >= self.thought_limits.max_branches:
                break
            node_id = str(entry.get("id") or "").strip()
            if not node_id:
                node_id = f"thought-{len(used_ids) + len(steps) + 1}"
            if node_id in used_ids:
                continue
            action = str(entry.get("action") or "").strip().lower()
            if action not in self.available_actions:
                continue
            question = (entry.get("question") or "").strip()
            if not question:
                continue
            parent_id = (entry.get("parent") or "root").strip() or "root"
            context_value = (entry.get("context") or "").strip() or None
            hints_value = entry.get("hints") or []
            hints = [str(hint).strip() for hint in hints_value if str(hint).strip()]
            max_results_raw = entry.get("max_results")
            try:
                max_results = int(max_results_raw) if max_results_raw is not None else None
            except (TypeError, ValueError):
                max_results = None
            justification = (entry.get("notes") or entry.get("reason") or "").strip()
            steps.append(
                ThoughtStepRequest(
                    node_id=node_id,
                    parent_id=parent_id,
                    action=action,
                    question=question,
                    context=context_value,
                    hints=hints,
                    max_results=max_results,
                    justification=justification,
                )
            )
        return steps

    def _execute_thought_step(
        self,
        step: ThoughtStepRequest,
        bundle: PreprocessBundle,
        depth: int,
        display_id: str,
    ) -> ThoughtStepResult:
        """Execute one planner node with retries and wrap the outcome in ``ThoughtStepResult``."""
        attempts = 0
        last_error = ""
        while attempts < self.max_agent_retries:
            attempts += 1
            try:
                success, summary, output = self._run_thought_action(step, bundle)
                return ThoughtStepResult(
                    node_id=step.node_id,
                    parent_id=step.parent_id or "root",
                    action=step.action,
                    depth=depth,
                    success=success,
                    summary=summary,
                    output=output,
                    justification=step.justification,
                    attempts=attempts,
                    display_id=display_id,
                )
            except LLMCommunicationError as exc:
                last_error = str(exc)

        fallback_output = {"error": last_error or "Agent unavailable"}
        failure_summary = (
            f"Agent failed after {attempts} attempt(s): {last_error}"
            if last_error
            else "Agent failed without diagnostic."
        )
        return ThoughtStepResult(
            node_id=step.node_id,
            parent_id=step.parent_id or "root",
            action=step.action,
            depth=depth,
            success=False,
            summary=failure_summary,
            output=fallback_output,
            justification=step.justification,
            attempts=attempts,
            display_id=display_id,
        )

    def _run_thought_action(
        self, step: ThoughtStepRequest, bundle: PreprocessBundle
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Delegate a planner directive to semantic or syntax agent and normalize the payload."""
        if step.action == "semantic" and self.semantic_agent:
            try:
                semantic_input = SemanticAgentInput(
                    question=step.question,
                    max_results=step.max_results or self.semantic_default_top_k,
                )
                sem_output = self.semantic_agent.handle(semantic_input)
                output = {
                    "question": sem_output.question,
                    "note": sem_output.note,
                    "hits": [asdict(hit) for hit in sem_output.hits],
                }
                summary = sem_output.note
                return True, summary, output
            except LLMCommunicationError:
                raise
            except Exception as exc:  # pragma: no cover - runtime guard
                output = {"error": str(exc)}
                summary = f"Semantic agent failed: {exc}"
                return False, summary, output

        if step.action == "syntax" and self.syntax_agent:
            try:
                syntax_input = SyntaxAgentInput(
                    question=step.question,
                    context=step.context or bundle.normalized_question,
                    hints=step.hints or None,
                )
                syntax_output = self.syntax_agent.handle(syntax_input)
                
                # Parse occurrences from command stdout for structured access
                # NOTE: We only collect raw data here. Compute agent handles deduplication/counting.
                occurrences = []
                unique_scripts = set()
                
                for cmd in syntax_output.commands:
                    if cmd.returncode == 0 and cmd.stdout:
                        stdout_stripped = cmd.stdout.strip()
                        
                        # Check if stdout is JSON (from python:scan_script_references)
                        if stdout_stripped.startswith("{"):
                            try:
                                parsed = json.loads(stdout_stripped)
                                # Extract structured data from scanner output
                                results = parsed.get("results", [])
                                for script_result in results:
                                    script_path = script_result.get("file_path", "")
                                    script_id = script_result.get("script_id", "")
                                    unique_scripts.add(script_path)
                                    
                                    for hit in script_result.get("hits", []):
                                        resolved_path = hit.get("resolved_path", "")
                                        occurrences.append({
                                            "script_id": script_id,
                                            "script_path": script_path,
                                            "line": hit.get("line", ""),
                                            "verb": hit.get("verb", ""),
                                            "raw": hit.get("raw", "")[:100],
                                            "resolved_path": resolved_path,
                                        })
                                continue  # Skip grep-style parsing
                            except json.JSONDecodeError:
                                pass  # Fall back to grep-style parsing
                        
                        # Grep-style parsing (path:line:content)
                        for line in stdout_stripped.splitlines():
                            if not line or line.startswith("--"):
                                continue
                            parts = line.split(":", 2)
                            if len(parts) >= 2:
                                script_path = parts[0]
                                line_num = parts[1] if len(parts) > 1 else ""
                                content = parts[2] if len(parts) > 2 else ""
                                
                                script_id = self._infer_id_from_path(script_path)
                                friendly_path = self.script_mapping.get(script_id, script_path) if script_id else script_path
                                
                                unique_scripts.add(friendly_path)
                                occurrences.append({
                                    "script_path": script_path,
                                    "resolved_path": friendly_path,
                                    "line": line_num,
                                    "content": content.strip()[:100],
                                })
                
                output = {
                    "question": syntax_output.question,
                    "summary": syntax_output.summary,
                    "commands": [asdict(cmd) for cmd in syntax_output.commands],
                    # Raw structured results - Thinker must use compute for deduplication/counting
                    "scripts_count": len(unique_scripts),
                    "occurrences_count": len(occurrences),
                    "occurrences_data": occurrences,  # Raw data for compute agent
                }
                summary = syntax_output.summary
                return True, summary, output
            except LLMCommunicationError:
                raise
            except Exception as exc:  # pragma: no cover - runtime guard
                output = {"error": str(exc)}
                summary = f"Syntax agent failed: {exc}"
                return False, summary, output

        output = {"error": f"Action '{step.action}' unavailable."}
        summary = output["error"]
        return False, summary, output

    def _enforce_minimum_steps(
        self,
        *,
        executed_steps: List[ThoughtStepResult],
        needed: int,
        bundle: PreprocessBundle,
        display_ids: Dict[str, str],
        child_counters: Dict[str, int],
        used_ids: Set[str],
        history: List[Dict[str, Any]],
        pending_router_actions: List[str],
    ) -> None:
        """Force extra executions so the number of nodes meets ``min_iterations``."""

        if needed <= 0 or not self.available_actions:
            return
        # Prioritise remaining router actions before repeating already used ones.
        priority = [action for action in pending_router_actions if action in self.available_actions]
        if not priority:
            priority = list(self.available_actions)
        actions: List[str] = []
        idx = 0
        while len(actions) < needed and priority:
            actions.append(priority[idx % len(priority)])
            idx += 1
        for action in actions:
            node_id = self._generate_node_id(used_ids, prefix="enforced")
            used_ids.add(node_id)
            display_id = self._assign_display_id(
                node_id=node_id,
                parent_id="root",
                display_ids=display_ids,
                child_counters=child_counters,
            )
            request = ThoughtStepRequest(
                node_id=node_id,
                parent_id="root",
                action=action,
                question=bundle.normalized_question,
                context=None,
                hints=[],
                max_results=None,
                justification="Auto-enforced step to satisfy min_iterations",
            )
            result = self._execute_thought_step(request, bundle, depth=1, display_id=display_id)
            executed_steps.append(result)
            history.append(
                {
                    "id": result.node_id,
                    "parent": result.parent_id,
                    "action": result.action,
                    "depth": result.depth,
                    "success": result.success,
                    "summary": result.summary,
                    "justification": result.justification,
                    "display_id": result.display_id,
                    "attempts": result.attempts,
                }
            )
            if action in pending_router_actions:
                pending_router_actions.remove(action)

    # ==========================================================================
    # THINKING-FIRST ARCHITECTURE - Iterative reasoning loop
    # ==========================================================================

    def _run_thinking_loop(
        self,
        question: str,
        context: str,
        bundle: PreprocessBundle,
    ) -> Tuple[WorkingMemory, Dict[str, Any]]:
        """Drive the thinking-first reasoning loop with Graph of Thoughts.
        
        This loop implements hierarchical exploration with splits and backtracking:
        1. Maintains persistent WorkingMemory with node tracking
        2. Shows the thinker LLM the full state including limits
        3. Supports SPLITS for parallel exploration (simulated via DFS)
        4. Forces BACKTRACK when max_depth reached
        5. Tracks iterations vs total_steps separately
        
        Returns:
            Tuple of (final WorkingMemory state, report dict for telemetry)
        """
        # Initialize working memory with config limits
        memory = WorkingMemory(
            max_iterations=self.thought_limits.max_iterations,
            min_iterations=self.thought_limits.min_iterations,
            max_depth=self.thought_limits.max_depth,
            max_branches=self.thought_limits.max_branches,
            max_total_steps=self.thought_limits.max_total_steps,
            confidence_threshold=0.8,
        )
        
        # Seed open questions from decomposition
        decomp = bundle.decomposition or {}
        sub_questions = decomp.get("sub_questions", [])
        if sub_questions:
            memory.open_questions = [
                q.get("question", q) if isinstance(q, dict) else str(q)
                for q in sub_questions
            ]
        else:
            memory.open_questions = [bundle.normalized_question]
        
        # Track execution for report
        last_observation: Optional[Dict[str, Any]] = None
        executed_steps: List[ThoughtStepResult] = []
        
        while True:
            # Increment iteration counter (Thinker decisions)
            memory.stats.iterations += 1
            memory.iteration = memory.stats.iterations
            
            # Check HARD stop conditions
            can_continue, stop_reason = memory.can_continue()
            if not can_continue:
                if self.verbose:
                    print(f"\n[Thinker] FORCED STOP: {stop_reason}")
                memory.forced_stop_reason = stop_reason
                break
            
            # Check if we MUST backtrack (max_depth reached)
            if memory.should_backtrack():
                if self.verbose:
                    print(f"\n[Auto-Backtrack] Max depth {memory.max_depth} reached, forcing backtrack")
                # Inject a backtrack action
                backtrack_result = self._handle_backtrack(
                    memory,
                    node_id=f"auto-backtrack-{memory.iteration}",
                    parent_id=memory.current_node_index,
                )
                executed_steps.append(backtrack_result)
                memory.stats.record_agent_call("backtrack", 1)
                last_observation = self._build_observation(backtrack_result)
                memory.raw_observations.append(last_observation)
                continue  # Re-enter loop with new depth
            
            # Ask the thinker for next action
            thinker_response = self._invoke_thinker(
                question=question,
                context=context,
                bundle=bundle,
                memory=memory,
                last_observation=last_observation,
            )
            
            if self.verbose:
                thinking_text = thinker_response.get('thinking', '(no thinking)')
                node_prefix = memory.current_node_index or "root"
                remaining_iter = memory.max_iterations - memory.stats.iterations
                remaining_steps = memory.max_total_steps - memory.stats.total_steps
                print(f"\n[Thinker] Iteration {memory.stats.iterations}/{memory.max_iterations} | Node: {node_prefix} | Depth: {memory.current_depth}/{memory.max_depth}")
                print(f"  Remaining: {remaining_iter} iterations, {remaining_steps} steps")
                if memory.split_stack:
                    pending = sum(len(s.pending_branches) for s in memory.split_stack)
                    print(f"  Active splits: {len(memory.split_stack)}, pending branches: {pending}")
                if memory.exploration_stack:
                    print(f"  Exploration path: {' → '.join(s.get('script', '?') for s in memory.exploration_stack)}")
                print(f"  Thinking:")
                for line in thinking_text.split('\n'):
                    print(f"    {line}")
                # Show decision with agent type
                decision_str = thinker_response.get('decision', 'unknown')
                action_spec = thinker_response.get('action', {})
                action_type = action_spec.get('type', '') if action_spec else ''
                if action_type:
                    print(f"  Decision: {decision_str} → {action_type}")
                else:
                    print(f"  Decision: {decision_str}")
            
            # Apply memory updates from thinker
            self._apply_memory_updates(memory, thinker_response.get("memory_updates", {}))
            
            # Check decision
            decision = thinker_response.get("decision", "stop")
            
            # Handle fail decision
            if decision == "fail":
                fail_reason = thinker_response.get("fail_reason", "Unable to provide reliable answer")
                if self.verbose:
                    print(f"  [Thinker FAILED]: {fail_reason}")
                memory.fail_reason = fail_reason
                break
            
            # Handle split decision
            if decision == "split":
                action_spec = thinker_response.get("action", {})
                if action_spec.get("type") != "split":
                    action_spec["type"] = "split"
                # Fall through to action execution
                decision = "act"
            
            if decision == "stop":
                # Validate: don't allow stop if last observation shows empty/failed results
                last_result_problematic = False
                if last_observation:
                    output = last_observation.get("output", {})
                    if isinstance(output, dict):
                        scripts_count = output.get("scripts_count", -1)
                        occurrences = output.get("occurrences", -1)
                        if scripts_count == 0 or occurrences == 0:
                            last_result_problematic = True
                        if output.get("error"):
                            last_result_problematic = True
                    if last_observation.get("success") == False:
                        last_result_problematic = True
                
                if last_result_problematic and memory.stats.iterations < memory.max_iterations:
                    if self.verbose:
                        print(f"  [WARNING] Empty/failed result - forcing retry")
                    decision = "act"
                    if not thinker_response.get("action"):
                        thinker_response["action"] = {
                            "type": "syntax",
                            "question": f"RETRY: {bundle.normalized_question}",
                        }
                elif memory.stats.iterations < memory.min_iterations:
                    if self.verbose:
                        print(f"  [min_iterations not reached - forcing continuation]")
                    decision = "act"
                    if not thinker_response.get("action"):
                        thinker_response["action"] = {"type": "syntax", "query": bundle.normalized_question}
                else:
                    if self.verbose:
                        print(f"  Stop reason: {thinker_response.get('stop_reason', 'confidence reached')}")
                    break
            
            # Execute the action
            action_spec = thinker_response.get("action", {})
            if not action_spec or not action_spec.get("type"):
                if memory.stats.iterations < memory.min_iterations:
                    action_spec = {"type": "syntax", "query": bundle.normalized_question}
                else:
                    if self.verbose:
                        print("  [Warning] No action specified, stopping")
                    break
            
            action_result = self._execute_thinker_action(action_spec, bundle, memory)
            executed_steps.append(action_result)
            
            # Record stats
            memory.stats.record_agent_call(action_result.action, action_result.attempts)
            
            # Handle retry vs new node
            if getattr(action_result, 'is_retry', False) and memory.current_node_index:
                # Retry: update existing node instead of creating a new one
                existing_node = memory.nodes.get(memory.current_node_index)
                if existing_node:
                    existing_node.attempts += 1
                    existing_node.success = action_result.success
                    existing_node.output = action_result.output
                    existing_node.summary = action_result.summary
                    node = existing_node
                    action_result.attempts = existing_node.attempts
                else:
                    # Fallback: create new node if existing not found
                    node = memory.add_node(
                        agent=action_result.action,
                        question=action_spec.get("question", ""),
                        description=action_result.summary[:50],
                        parent_index=memory.current_node_index if memory.current_node_index else "",
                    )
            else:
                # New action: create a new node
                node = memory.add_node(
                    agent=action_result.action,
                    question=action_spec.get("question", ""),
                    description=action_result.summary[:50],
                    parent_index=memory.current_node_index if memory.current_node_index else "",
                )
            node.success = action_result.success
            node.attempts = action_result.attempts
            node.output = action_result.output
            node.summary = action_result.summary
            
            # Update current_node_index to this node (next action will be a child)
            # This creates a chain: syntax(depth=1) -> compute(depth=2) -> etc.
            memory.current_node_index = node.index
            memory.current_depth = node.depth  # Sync depth for accurate tracking
            
            # Update display_id and depth with actual node values
            action_result.display_id = node.index
            action_result.depth = node.depth
            
            # Record the observation
            observation = self._build_observation(action_result)
            memory.raw_observations.append(observation)
            memory.actions_taken.append({
                "iteration": memory.stats.iterations,
                "node_index": node.index,
                "action": action_spec,
                "success": action_result.success,
            })
            last_observation = observation
            
            # Extract patterns from observation
            self._extract_patterns_from_observation(memory, observation)
            
            if self.verbose:
                # Show full action result with complete output (transparent like other stages)
                self._print_thought_step_summary(action_result.serialize(), memory.stats.iterations)
                self._print_memory_state(memory)
        
        # Build comprehensive report
        status = "failed" if memory.fail_reason else "completed"
        if memory.forced_stop_reason:
            status = "forced_stop"
        
        report = {
            "status": status,
            "stats": {
                "iterations": memory.stats.iterations,
                "total_steps": memory.stats.total_steps,
                "max_depth_reached": memory.stats.max_depth_reached,
                "retries": memory.stats.retries,
                "splits_created": memory.stats.splits_created,
                "branches_explored": memory.stats.branches_explored,
                "agents_called": memory.stats.agents_called,
            },
            "limits": {
                "max_iterations": memory.max_iterations,
                "max_total_steps": memory.max_total_steps,
                "max_depth": memory.max_depth,
                "max_branches": memory.max_branches,
            },
            "final_confidence": memory.get_best_hypothesis().confidence if memory.get_best_hypothesis() else 0.0,
            "hypotheses_count": len(memory.hypotheses),
            "patterns_discovered": list(memory.patterns),
            "scripts_of_interest": [s[0] for s in memory.scripts_of_interest],
            "nodes_count": len(memory.nodes),
            "thought_tree": memory.render_tree(),
            "steps": [step.serialize() for step in executed_steps],
            "working_memory": memory.serialize(),
        }
        if memory.fail_reason:
            report["fail_reason"] = memory.fail_reason
        if memory.forced_stop_reason:
            report["forced_stop_reason"] = memory.forced_stop_reason
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GRAPH OF THOUGHTS - SUMMARY")
            print(f"{'='*60}")
            print(f"Status: {status}")
            print(f"Iterations: {memory.stats.iterations} / {memory.max_iterations}")
            print(f"Total Steps: {memory.stats.total_steps} / {memory.max_total_steps}")
            print(f"Max Depth Reached: {memory.stats.max_depth_reached} / {memory.max_depth}")
            if memory.stats.retries > 0:
                print(f"Retries: {memory.stats.retries}")
            if memory.stats.splits_created > 0:
                print(f"Splits Created: {memory.stats.splits_created}")
                print(f"Branches Explored: {memory.stats.branches_explored}")
            print(f"{'='*60}")
        
        return memory, report
    
    def _print_memory_state(self, memory: WorkingMemory) -> None:
        """Print current memory state for verbose mode."""
        print(f"  [Memory State]")
        print(f"    Hypotheses ({len(memory.hypotheses)}):")
        if memory.hypotheses:
            for h in memory.hypotheses:
                print(f"      • [{h.confidence:.2f}] {h.claim}")
        else:
            print("      (empty)")
        print(f"    Patterns ({len(memory.patterns)}):")
        if memory.patterns:
            for p in list(memory.patterns)[:5]:
                print(f"      • {p}")
            if len(memory.patterns) > 5:
                print(f"      ... and {len(memory.patterns) - 5} more")
        else:
            print("      (empty)")
        print(f"    Scripts ({len(memory.scripts_of_interest)}):")
        if memory.scripts_of_interest:
            for script_id, reason in memory.scripts_of_interest[:5]:
                print(f"      • {script_id} - {reason}")
            if len(memory.scripts_of_interest) > 5:
                print(f"      ... and {len(memory.scripts_of_interest) - 5} more")
        else:
            print("      (empty)")

    def _invoke_thinker(
        self,
        question: str,
        context: str,
        bundle: PreprocessBundle,
        memory: WorkingMemory,
        last_observation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the thinker LLM with full state visibility."""
        if not self.thought_agent:
            return {"decision": "stop", "stop_reason": "Thinker unavailable"}
        
        # Format exploration stack for display
        exploration_stack_str = "empty (at root level)"
        if memory.exploration_stack:
            stack_items = [f"→ {item.get('script', 'unknown')}" for item in memory.exploration_stack]
            exploration_stack_str = " ".join(stack_items)
        
        # Calculate remaining resources
        remaining_iterations = memory.max_iterations - memory.iteration
        remaining_steps = memory.max_total_steps - memory.stats.total_steps
        
        prompt = DEFAULT_THINKER_PROMPT.format(
            working_memory=json.dumps(memory.serialize(), indent=2, ensure_ascii=False),
            question=question,
            normalized_question=bundle.normalized_question,
            clarifier=json.dumps(bundle.clarifier or {}, ensure_ascii=False),
            decomposition=json.dumps(bundle.decomposition or {}, ensure_ascii=False),
            router=json.dumps(bundle.router or {}, ensure_ascii=False),
            last_observation=json.dumps(last_observation or {"note": "First iteration, no observation yet"}, indent=2, ensure_ascii=False),
            current_depth=memory.current_depth,
            max_depth=memory.max_depth,
            max_iterations=memory.max_iterations,
            remaining_iterations=remaining_iterations,
            max_total_steps=memory.max_total_steps,
            remaining_steps=remaining_steps,
            max_branches=memory.max_branches,
            max_agent_retries=self.max_agent_retries,
            current_node=memory.current_node_index or "root",
            active_splits=len(memory.split_stack),
            pending_branches=sum(len(s.pending_branches) for s in memory.split_stack),
            exploration_stack=exploration_stack_str,
        )
        
        try:
            raw = self.thought_agent.generate_response(prompt=prompt)
            return parse_llm_json(raw)
        except Exception as exc:
            if self.verbose:
                print(f"  [Error] Thinker failed: {exc}")
            # Return act with retry instead of stopping
            return {
                "decision": "act",
                "thinking": f"Previous response parsing failed: {exc}. Retrying with simpler approach.",
                "action": {
                    "type": "syntax",
                    "question": "Retry: find all import statements in .nvn files",
                },
                "memory_updates": {},
            }

    def _apply_memory_updates(
        self,
        memory: WorkingMemory,
        updates: Dict[str, Any],
    ) -> None:
        """Apply thinker-suggested updates to working memory."""
        if not updates:
            return
        
        # Add new hypotheses
        for h in updates.get("new_hypotheses", []):
            if h.get("claim"):
                memory.hypotheses.append(Hypothesis(
                    claim=h["claim"],
                    confidence=float(h.get("confidence", 0.5)),
                    evidence=h.get("evidence", []),
                    source_iteration=memory.iteration,
                ))
        
        # Update existing hypothesis confidences
        for update in updates.get("updated_confidences", []):
            claim = update.get("claim", "")
            for h in memory.hypotheses:
                if h.claim == claim:
                    h.confidence = float(update.get("new_confidence", h.confidence))
                    break
        
        # Add patterns
        for pattern in updates.get("new_patterns", []):
            if pattern:
                memory.patterns.add(pattern)
        
        # Add scripts of interest
        for script in updates.get("new_scripts_of_interest", []):
            if script.get("script"):
                memory.scripts_of_interest.append((script["script"], script.get("reason", "")))
        
        # Resolve questions
        resolved = set(updates.get("resolved_questions", []))
        memory.open_questions = [q for q in memory.open_questions if q not in resolved]
        
        # Add new questions
        for q in updates.get("new_questions", []):
            if q and q not in memory.open_questions:
                memory.open_questions.append(q)
        
        # Add dead ends
        for de in updates.get("new_dead_ends", []):
            if de.get("path"):
                memory.dead_ends.append((de["path"], de.get("reason", "")))

    def _execute_thinker_action(
        self,
        action_spec: Dict[str, Any],
        bundle: PreprocessBundle,
        memory: WorkingMemory,
    ) -> ThoughtStepResult:
        """Execute a single action specified by the thinker.
        
        Supports actions:
        - semantic: Semantic search across all scripts
        - syntax: Grep/regex search across scripts
        - read: Read content of a specific script (increases depth)
        - dive: Focus subsequent searches on a specific script (increases depth)
        - backtrack: Return to parent exploration level (decreases depth)
        - retry: Re-execute previous action type with corrected parameters
        """
        action_type = action_spec.get("type", "").lower()
        original_action_type = action_type  # Keep track for retry detection
        question = action_spec.get("question", bundle.normalized_question)
        context_val = action_spec.get("context")
        hints = action_spec.get("hints", [])
        target_script = action_spec.get("target_script", "")
        correction_feedback = action_spec.get("correction_feedback", "")
        
        # Handle retry - convert to the actual action type with correction context
        is_retry = False
        if action_type == "retry":
            is_retry = True
            # Get the last action from memory and retry it with correction
            if memory.actions_taken:
                last_action = memory.actions_taken[-1].get("action", {})
                action_type = last_action.get("type", "syntax")
                # Add correction context to help the agent
                if correction_feedback:
                    context_val = f"CORRECTION NEEDED: {correction_feedback}. Previous attempt was incorrect."
                    # Append to question for clarity
                    question = f"{question} [IMPORTANT: {correction_feedback}]"
            else:
                action_type = "syntax"  # Default fallback
        
        node_id = f"think-{memory.iteration}-d{memory.current_depth}"
        parent_id = memory.last_node_id  # Use tracked parent for tree linking
        
        # Handle special depth-first exploration actions
        if action_type == "backtrack":
            result = self._handle_backtrack(memory, node_id, parent_id)
            memory.last_node_id = node_id  # Update for next action
            return result
        
        if action_type == "dive":
            result = self._handle_dive(memory, target_script, node_id, parent_id, action_spec)
            memory.last_node_id = node_id  # Update for next action
            return result
        
        if action_type == "read":
            result = self._handle_read_script(memory, target_script, question, node_id, parent_id)
            memory.last_node_id = node_id  # Update for next action
            return result
        
        if action_type == "compute":
            result = self._handle_compute(memory, action_spec, node_id, parent_id)
            memory.last_node_id = node_id  # Update for next action
            return result
        
        if action_type == "split":
            result = self._handle_split(memory, action_spec, node_id, parent_id)
            memory.last_node_id = node_id  # Update for next action
            return result
        
        # For semantic/syntax, scope to current dive target if we're deep in exploration
        if memory.exploration_stack and memory.current_depth > 0:
            current_focus = memory.exploration_stack[-1]
            focus_script = current_focus.get("script", "")
            if focus_script and action_type == "syntax":
                # Scope the syntax search to the current script
                context_val = f"Focus search on script: {focus_script}"
                if focus_script not in question:
                    question = f"{question} (in script {focus_script})"
        
        step = ThoughtStepRequest(
            node_id=node_id,
            parent_id=parent_id,
            action=action_type,
            question=question,
            context=context_val,
            hints=hints,
            max_results=None,
            justification=action_spec.get("rationale", ""),
        )
        
        result = self._execute_thought_step(step, bundle, depth=memory.current_depth + 1, display_id=str(memory.iteration))
        
        # Mark if this was a retry (for node management in thinking loop)
        result.is_retry = is_retry
        
        memory.last_node_id = node_id  # Update for next action
        return result
    
    def _resolve_compute_data(self, data: Any, memory: WorkingMemory) -> Any:
        """Resolve data references for compute agent.
        
        If data is a string that references observation data (like "occurrences_data from last_observation"),
        extract the actual data from memory.raw_observations.
        
        Common patterns:
        - "occurrences_data from last_observation" → get occurrences_data from last observation
        - "last_observation.occurrences_data" → same
        - "results" or "occurrences_data" → look in last observation
        - Already structured data → return as-is
        """
        # If data is already structured (list, dict, etc.), return as-is
        if isinstance(data, (list, dict)) and data:
            return data
        
        # If data is empty or not a string, try to find data in last observation
        if not data or not isinstance(data, str):
            # Auto-resolve from last observation
            if memory.raw_observations:
                last_obs = memory.raw_observations[-1]
                # Priority: occurrences_data (from syntax), computed_result (from compute), hits (from semantic)
                if "occurrences_data" in last_obs:
                    return last_obs["occurrences_data"]
                if "computed_result" in last_obs:
                    return last_obs["computed_result"]
                if "hits" in last_obs:
                    return last_obs["hits"]
            return data
        
        # Parse string references
        data_lower = data.lower().strip()
        
        # Handle "last_observation" references
        if "last_observation" in data_lower or "last observation" in data_lower:
            if memory.raw_observations:
                last_obs = memory.raw_observations[-1]
                # Extract specific field if mentioned
                if "occurrences_data" in data_lower or "occurrences" in data_lower:
                    return last_obs.get("occurrences_data", [])
                if "computed_result" in data_lower or "result" in data_lower:
                    return last_obs.get("computed_result", [])
                if "hits" in data_lower:
                    return last_obs.get("hits", [])
                # Return the whole output
                return last_obs.get("occurrences_data") or last_obs.get("computed_result") or last_obs.get("hits") or []
        
        # Handle simple field references like "occurrences_data", "results"
        if data_lower in ("occurrences_data", "occurrences", "results", "data"):
            if memory.raw_observations:
                last_obs = memory.raw_observations[-1]
                return last_obs.get("occurrences_data") or last_obs.get("computed_result") or []
        
        # Handle "node X" references - find data from specific node
        import re
        node_match = re.search(r"node\s*[#]?(\d+)", data_lower)
        if node_match:
            target_index = node_match.group(1)
            for obs in reversed(memory.raw_observations):
                if obs.get("display_id") == target_index or obs.get("node_id", "").endswith(target_index):
                    return obs.get("occurrences_data") or obs.get("computed_result") or obs.get("hits") or []
        
        # If it's a short string that looks like a reference, try auto-resolve
        if len(data) < 100 and ("from" in data_lower or "observation" in data_lower or "output" in data_lower):
            if memory.raw_observations:
                last_obs = memory.raw_observations[-1]
                return last_obs.get("occurrences_data") or last_obs.get("computed_result") or last_obs.get("hits") or []
        
        # Return original data if no resolution possible
        return data
    
    def _handle_backtrack(self, memory: WorkingMemory, node_id: str, parent_id: str) -> ThoughtStepResult:
        """Handle backtrack action - return to parent exploration level.
        
        Backtracking uses the node index hierarchy:
        - From node "11" → go to node "1" (parent)
        - From node "1" → check for splits or go to root
        
        If we're at a split point with pending branches, this will:
        1. Mark current branch as explored
        2. Move to the next pending branch
        3. Set up exploration for that branch
        """
        current_index = memory.current_node_index
        
        # Can we backtrack via node hierarchy?
        if current_index and len(current_index) > 1:
            # Go to parent node (e.g., "11" → "1", "121" → "12")
            parent_index = current_index[:-1]
            
            # Check if there's a split at this level we should explore
            split = memory.backtrack_to_last_split()
            if split and split.pending_branches:
                # Move to next pending branch of the split
                next_branch = split.next_branch()
                if next_branch:
                    split.mark_explored(next_branch)
                    memory.current_node_index = next_branch
                    memory.current_depth = len(next_branch)
                    
                    branch_node = memory.nodes.get(next_branch)
                    branch_desc = branch_node.description if branch_node else f"branch {next_branch}"
                    
                    return ThoughtStepResult(
                        node_id=node_id,
                        parent_id=parent_id,
                        action="backtrack",
                        depth=memory.current_depth,
                        success=True,
                        summary=f"Backtracked to split, now exploring branch {next_branch}: {branch_desc}",
                        output={
                            "from_split": True,
                            "next_branch": next_branch,
                            "remaining_branches": split.pending_branches,
                            "new_depth": memory.current_depth,
                            "action_hint": "Create a new action to explore this branch",
                        },
                    )
            
            # No split, just go to parent
            memory.current_node_index = parent_index
            memory.current_depth = len(parent_index)
            
            parent_node = memory.nodes.get(parent_index)
            parent_desc = parent_node.agent if parent_node else "parent"
            
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="backtrack",
                depth=memory.current_depth,
                success=True,
                summary=f"Backtracked to node {parent_index} ({parent_desc}). You can now split to explore alternatives.",
                output={
                    "backtracked_to": parent_index,
                    "parent_agent": parent_desc,
                    "new_depth": memory.current_depth,
                    "action_hint": "Consider creating a split to explore alternative approaches",
                },
            )
        
        # We're at root level (depth 0 or 1 with single-digit index)
        # Check if there's a split with pending branches
        split = memory.backtrack_to_last_split()
        if split and split.pending_branches:
            next_branch = split.next_branch()
            if next_branch:
                split.mark_explored(next_branch)
                memory.current_node_index = next_branch
                memory.current_depth = len(next_branch)
                
                branch_node = memory.nodes.get(next_branch)
                branch_desc = branch_node.description if branch_node else f"branch {next_branch}"
                
                return ThoughtStepResult(
                    node_id=node_id,
                    parent_id=parent_id,
                    action="backtrack",
                    depth=memory.current_depth,
                    success=True,
                    summary=f"Split backtrack: Now exploring branch {next_branch}: {branch_desc}",
                    output={
                        "from_split": True,
                        "next_branch": next_branch,
                        "remaining_branches": split.pending_branches,
                        "new_depth": memory.current_depth,
                    },
                )
        
        # Pop from exploration_stack if it has content (legacy behavior)
        if memory.exploration_stack:
            popped = memory.exploration_stack.pop()
            memory.current_depth = max(0, memory.current_depth - 1)
            
            was_split_branch = popped.get("is_split_branch", False)
            if was_split_branch:
                split = memory.backtrack_to_last_split()
                if split and split.pending_branches:
                    next_branch = split.pending_branches[0]
                    return ThoughtStepResult(
                        node_id=node_id,
                        parent_id=parent_id,
                        action="backtrack",
                        depth=memory.current_depth,
                        success=True,
                        summary=f"Backtracked from split branch. Next pending branch: {next_branch}. Call backtrack again to explore it.",
                        output={
                            "backtracked_from": popped,
                            "new_depth": memory.current_depth,
                            "has_pending_split": True,
                            "next_pending_branch": next_branch,
                            "pending_count": len(split.pending_branches),
                        },
                    )
            
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="backtrack",
                depth=memory.current_depth,
                success=True,
                summary=f"Backtracked from {popped.get('script', 'unknown')}. Now at depth {memory.current_depth}.",
                output={
                    "backtracked_from": popped,
                    "new_depth": memory.current_depth,
                    "remaining_stack": memory.exploration_stack,
                },
            )
        
        # Truly at root with no way to backtrack
        return ThoughtStepResult(
            node_id=node_id,
            parent_id=parent_id,
            action="backtrack",
            depth=memory.current_depth,
            success=False,
            summary="Already at root level. Consider creating a split to explore alternatives.",
            output={
                "error": "At root level",
                "action_hint": "Use split action to explore alternative approaches",
            },
        )
    
    def _handle_dive(
        self,
        memory: WorkingMemory,
        target_script: str,
        node_id: str,
        parent_id: str,
        action_spec: Dict[str, Any],
    ) -> ThoughtStepResult:
        """Handle dive action - go deeper into a specific script."""
        if memory.current_depth >= memory.max_depth:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="dive",
                depth=memory.current_depth,
                success=False,
                summary=f"Cannot dive: max depth {memory.max_depth} reached",
                output={"error": f"Max depth {memory.max_depth} reached"},
            )
        
        if not target_script:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="dive",
                depth=memory.current_depth,
                success=False,
                summary="Cannot dive: no target_script specified",
                output={"error": "No target_script specified"},
            )
        
        # Push to stack and increase depth
        memory.exploration_stack.append({
            "script": target_script,
            "reason": action_spec.get("rationale", "Exploration"),
            "depth": memory.current_depth + 1,
        })
        memory.current_depth += 1
        memory.scripts_explored.add(target_script)
        
        return ThoughtStepResult(
            node_id=node_id,
            parent_id=parent_id,
            action="dive",
            depth=memory.current_depth,
            success=True,
            summary=f"Dived into {target_script}. Now at depth {memory.current_depth}. Subsequent actions will focus on this script.",
            output={
                "target_script": target_script,
                "new_depth": memory.current_depth,
                "exploration_stack": memory.exploration_stack,
            },
        )
    
    def _handle_read_script(
        self,
        memory: WorkingMemory,
        target_script: str,
        question: str,
        node_id: str,
        parent_id: str,
    ) -> ThoughtStepResult:
        """Handle read action - read content of a specific script."""
        if not target_script:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="read",
                depth=memory.current_depth,
                success=False,
                summary="Cannot read: no target_script specified",
                output={"error": "No target_script specified"},
            )
        
        # Try to resolve script to file path
        script_path = self._resolve_script_path(target_script)
        if not script_path or not script_path.exists():
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="read",
                depth=memory.current_depth,
                success=False,
                summary=f"Script not found: {target_script}",
                output={"error": f"Script not found: {target_script}"},
            )
        
        try:
            content = script_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            # If we have a question, search for relevant sections
            relevant_sections = []
            if question:
                search_terms = self._extract_search_terms(question)
                for i, line in enumerate(lines):
                    for term in search_terms:
                        if term.lower() in line.lower():
                            start = max(0, i - 3)
                            end = min(len(lines), i + 4)
                            section = "\n".join(f"{start+j+1}: {lines[start+j]}" for j in range(end - start))
                            relevant_sections.append({
                                "line": i + 1,
                                "match_term": term,
                                "context": section,
                            })
                            break
            
            # Track this as a dive
            if target_script not in memory.scripts_explored:
                memory.scripts_explored.add(target_script)
                memory.exploration_stack.append({
                    "script": target_script,
                    "reason": "Read for analysis",
                    "depth": memory.current_depth + 1,
                })
                memory.current_depth += 1
            
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="read",
                depth=memory.current_depth,
                success=True,
                summary=f"Read {target_script}: {len(lines)} lines, {len(relevant_sections)} relevant sections found",
                output={
                    "script": target_script,
                    "total_lines": len(lines),
                    "relevant_sections": relevant_sections[:10],  # Limit to avoid explosion
                    "first_50_lines": "\n".join(lines[:50]),
                },
            )
        except Exception as exc:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="read",
                depth=memory.current_depth,
                success=False,
                summary=f"Error reading {target_script}: {exc}",
                output={"error": str(exc)},
            )
    
    def _handle_compute(
        self,
        memory: WorkingMemory,
        action_spec: Dict[str, Any],
        node_id: str,
        parent_id: str,
    ) -> ThoughtStepResult:
        """Handle compute action - delegate computation to code agent."""
        if not self.code_agent:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="compute",
                depth=memory.current_depth,
                success=False,
                summary="Code agent not available",
                output={"error": "Code agent is not configured"},
            )
        
        question = action_spec.get("question", "")
        data = action_spec.get("data", "")
        context = action_spec.get("context", "")
        
        # Resolve data references - if data is a string reference like "occurrences_data from last_observation",
        # extract the actual data from memory.raw_observations
        resolved_data = self._resolve_compute_data(data, memory)
        
        if not question:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="compute",
                depth=memory.current_depth,
                success=False,
                summary="No computation task specified",
                output={"error": "No question/task specified for computation"},
            )
        
        try:
            # Build context from action spec and memory
            full_context = context
            if resolved_data:
                data_preview_for_context = str(resolved_data)[:500] + "..." if len(str(resolved_data)) > 500 else str(resolved_data)
                full_context = f"{context}\n\nData to process:\n{data_preview_for_context}" if context else f"Data to process:\n{data_preview_for_context}"
            
            # Execute the computation with resolved data
            input_data = CodeAgentInput(
                task_description=question,
                data=resolved_data,  # Pass actual data, not string reference
                context=full_context,
            )
            
            if self.verbose:
                print(f"  [COMPUTE] Task: {question[:80]}...")
                if resolved_data:
                    data_len = len(resolved_data) if isinstance(resolved_data, (list, dict)) else len(str(resolved_data))
                    print(f"  [COMPUTE] Data: {type(resolved_data).__name__} with {data_len} items/chars")
            
            result = self.code_agent.run(input_data)
            
            if result.success:
                return ThoughtStepResult(
                    node_id=node_id,
                    parent_id=parent_id,
                    action="compute",
                    depth=memory.current_depth,
                    success=True,
                    summary=f"Computation successful: {result.result}",
                    output={
                        "result": result.result,
                        "code_executed": result.code_executed,
                        "explanation": result.explanation,
                        "execution_time_ms": result.execution_time_ms,
                    },
                )
            else:
                return ThoughtStepResult(
                    node_id=node_id,
                    parent_id=parent_id,
                    action="compute",
                    depth=memory.current_depth,
                    success=False,
                    summary=f"Computation failed: {result.error}",
                    output={
                        "error": result.error,
                        "code_attempted": result.code_executed,
                    },
                )
                
        except Exception as exc:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="compute",
                depth=memory.current_depth,
                success=False,
                summary=f"Compute error: {exc}",
                output={"error": str(exc)},
            )
    
    def _handle_split(
        self,
        memory: WorkingMemory,
        action_spec: Dict[str, Any],
        node_id: str,
        parent_id: str,
    ) -> ThoughtStepResult:
        """Handle split action - create parallel branches for simultaneous exploration.
        
        Creates a SplitPoint and returns info about the first branch to explore.
        The system will simulate parallelism via DFS - exploring one branch fully
        before backtracking to explore the next.
        """
        branches = action_spec.get("branches", [])
        
        if not branches:
            return ThoughtStepResult(
                node_id=node_id,
                parent_id=parent_id,
                action="split",
                depth=memory.current_depth,
                success=False,
                summary="Cannot split: no branches specified",
                output={"error": "No branches specified in action.branches"},
            )
        
        # Limit branches to max_branches
        if len(branches) > memory.max_branches:
            branches = branches[:memory.max_branches]
        
        # Create node indices for each branch
        branch_indices = []
        for i, branch in enumerate(branches, start=1):
            # Generate child index from current node
            branch_idx = memory.get_next_child_index(memory.current_node_index)
            branch_indices.append(branch_idx)
            
            # Create a ThoughtNode for the branch
            node = ThoughtNode(
                index=branch_idx,
                parent_index=memory.current_node_index,
                agent="split",
                description=branch.get("description", f"Branch {i}"),
                question=branch.get("query", ""),
                depth=len(branch_idx),
            )
            memory.nodes[branch_idx] = node
        
        # Create the split point
        split = memory.create_split(branch_indices)
        
        # The first branch becomes our current exploration target
        first_branch = branches[0]
        first_index = branch_indices[0]
        
        # Move to the first branch
        memory.current_node_index = first_index
        memory.exploration_stack.append({
            "script": first_branch.get("description", "split branch"),
            "reason": first_branch.get("query", ""),
            "depth": memory.current_depth + 1,
            "is_split_branch": True,
            "branch_index": first_index,
        })
        memory.current_depth += 1
        
        # Mark first branch as being explored (will be marked complete on backtrack)
        split.pending_branches.remove(first_index)
        split.explored_branches.append(first_index)
        
        return ThoughtStepResult(
            node_id=node_id,
            parent_id=parent_id,
            action="split",
            depth=memory.current_depth,
            success=True,
            summary=f"Created split with {len(branches)} branches. Now exploring branch {first_index}: {first_branch.get('description', '')}",
            output={
                "split_created": True,
                "total_branches": len(branches),
                "branch_indices": branch_indices,
                "current_branch": first_index,
                "pending_branches": split.pending_branches,
                "first_branch_action": first_branch.get("action_type", "syntax"),
                "first_branch_query": first_branch.get("query", ""),
            },
        )
    
    def _resolve_script_path(self, script_ref: str) -> Optional[Path]:
        """Resolve a script reference to a file path."""
        # If it's a numeric ID, look it up in mapping
        if script_ref.isdigit():
            script_id = script_ref
            if script_id in self.script_mapping:
                friendly_path = self.script_mapping[script_id]
                # Find the actual file
                for ext in (".nvn",):
                    candidate = self.script_root / f"{script_id}{ext}"
                    if candidate.exists():
                        return candidate
        
        # If it's a path, try to find the script
        for ext in (".nvn",):
            # Direct path
            candidate = Path(script_ref)
            if candidate.exists():
                return candidate
            # Under script_mirror
            candidate = self.script_root / f"{script_ref}{ext}"
            if candidate.exists():
                return candidate
        
        # Search by name pattern
        for file in self.script_root.glob("*.nvn"):
            if script_ref in file.stem or script_ref in str(file):
                return file
        
        return None
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract meaningful search terms from a question."""
        # Common English terms for code search
        common_terms = []
        
        # Extract quoted strings
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        common_terms.extend(quoted)
        
        # Extract CamelCase/PascalCase words
        camel = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text)
        common_terms.extend(camel)
        
        # Extract words that look like identifiers
        identifiers = re.findall(r"\b([A-Z][a-z]*[A-Z][a-z]*[A-Za-z]*)\b", text)
        common_terms.extend(identifiers)
        
        # Known patterns
        known = ["BestSeller", "TopSeller", "Rank", "Sales", "Items", "Catalog"]
        for k in known:
            if k.lower() in text.lower():
                common_terms.append(k)
        
        return list(set(common_terms))

    def _build_observation(self, result: ThoughtStepResult) -> Dict[str, Any]:
        """Convert ThoughtStepResult to an observation dict for the thinker.
        
        CRITICAL: The observation must contain the actual results, not just summaries,
        so the Thinker can analyze and decide what to do next.
        """
        obs = {
            "action": result.action,
            "success": result.success,
            "summary": result.summary,
            "node_id": result.node_id,
            "display_id": result.display_id,
            "depth": result.depth,
            "attempts": result.attempts,
        }
        
        # Include the full output with results
        output = result.output or {}
        
        # For syntax agent: include raw occurrences for compute agent to process
        if "occurrences_count" in output:
            obs["occurrences_count"] = output.get("occurrences_count", 0)
            obs["scripts_count"] = output.get("scripts_count", 0)
            
            # Include raw occurrences data for compute agent to process
            # Thinker MUST use compute to extract/count/deduplicate from this data
            occurrences = output.get("occurrences_data", [])
            if isinstance(occurrences, list) and occurrences:
                obs["occurrences_data"] = occurrences  # Full data for compute
                obs["data_note"] = "Use compute agent to extract unique values, count, or process this data"
        
        # For compute agent: include the computed result
        if "result" in output:
            obs["computed_result"] = output["result"]
        if "code_executed" in output:
            obs["code_executed"] = output["code_executed"]
        if "explanation" in output:
            obs["explanation"] = output["explanation"]
            
        # For semantic agent: include hits
        if "hits" in output:
            hits = output["hits"]
            if isinstance(hits, list):
                obs["hits"] = hits[:10]  # Limit to 10 hits
                obs["hits_count"] = len(hits)
        
        # For read/dive: include relevant sections
        if "relevant_sections" in output:
            obs["relevant_sections"] = output["relevant_sections"][:5]
        if "first_50_lines" in output:
            obs["content_preview"] = output["first_50_lines"][:500]
            
        # Include errors
        if "error" in output:
            obs["error"] = str(output["error"])
            
        # Include commands for syntax (so Thinker can see what was searched)
        if "commands" in output:
            commands = output["commands"]
            if isinstance(commands, list):
                obs["commands_executed"] = [
                    {"command": c.get("command", ""), "returncode": c.get("returncode", -1)}
                    for c in commands[:3]
                ]
            
        return obs

    def _truncate_output(self, output: Dict[str, Any], max_chars: int = 1500) -> Dict[str, Any]:
        """Truncate large outputs to keep prompts manageable and avoid JSON issues."""
        result = {}
        for key, value in output.items():
            if isinstance(value, str):
                # Clean special characters that can break JSON parsing
                cleaned = value.replace('\n', ' ').replace('\r', '')
                if len(cleaned) > max_chars:
                    result[key] = cleaned[:max_chars] + f"... [truncated, {len(value)} chars total]"
                else:
                    result[key] = cleaned
            elif isinstance(value, list) and len(value) > 5:
                # Take first 5 items only
                result[key] = value[:5]
                result[f"{key}_count"] = len(value)
                result[f"{key}_note"] = f"Showing 5 of {len(value)} items"
            else:
                result[key] = value
        return result

    def _extract_patterns_from_observation(
        self,
        memory: WorkingMemory,
        observation: Dict[str, Any],
    ) -> None:
        """Extract meaningful patterns from agent outputs.
        
        This is where we do "genuine reasoning" - identifying:
        - Script paths that appear frequently
        - Table/column names mentioned
        - Quoted strings that might be significant
        - Ion file paths
        """
        output = observation.get("output_preview", {})
        text_to_scan = json.dumps(output, ensure_ascii=False)
        
        # Extract quoted strings (table names, paths, etc.)
        # Use a cleaner regex that avoids escaped quotes
        quoted_patterns = re.findall(r'"(/[^"]{3,60})"', text_to_scan)
        for pattern in quoted_patterns:
            # Clean up the pattern
            cleaned = pattern.replace("\\", "").strip()
            # Filter out common noise and overly long patterns
            if cleaned and len(cleaned) <= 50:
                noise_words = {"true", "false", "null", "none", "error", "success", "command", "question"}
                if cleaned.lower() not in noise_words:
                    # Only keep meaningful patterns: paths, table names, .ion files
                    if cleaned.startswith("/") and ".ion" in cleaned:
                        memory.patterns.add(cleaned)
                    elif cleaned.startswith("show "):
                        memory.patterns.add(cleaned)
        
        # Extract Ion file paths specifically
        ion_paths = re.findall(r'(/[A-Za-z0-9_./\-]+\.ion)', text_to_scan)
        for path in ion_paths[:5]:
            cleaned = path.replace("\\", "").strip()
            if cleaned:
                memory.patterns.add(cleaned)
        
        # Extract script paths from hits
        hits = output.get("hits", [])
        if isinstance(hits, list):
            for hit in hits[:5]:  # Limit to avoid explosion
                if isinstance(hit, dict):
                    script = hit.get("script_id") or hit.get("source", "")
                    if script and script not in [s[0] for s in memory.scripts_of_interest]:
                        # Check if already tracked
                        memory.scripts_of_interest.append((script, "discovered via semantic search"))
        
        # Extract from command outputs
        commands = output.get("commands", [])
        if isinstance(commands, list):
            for cmd in commands:
                if isinstance(cmd, dict):
                    stdout = cmd.get("stdout", "")
                    # Extract file paths from grep output
                    path_matches = re.findall(r"([A-Za-z0-9_./-]+\.nvn)", stdout)
                    for path in path_matches[:3]:
                        if path not in [s[0] for s in memory.scripts_of_interest]:
                            memory.scripts_of_interest.append((path, "discovered via grep"))

    # ==========================================================================
    # END THINKING-FIRST ARCHITECTURE
    # ==========================================================================

    @staticmethod
    def _generate_node_id(used_ids: Set[str], prefix: str) -> str:
        counter = 1
        candidate = f"{prefix}-{counter}"
        while candidate in used_ids:
            counter += 1
            candidate = f"{prefix}-{counter}"
        return candidate

    def _print_stage_block(self, stage: StageExecution, index: int) -> None:
        """Pretty-print a single preprocess stage for verbose sessions."""
        separator = "-" * 56
        print(f"\n{separator}")
        print(
            f"[Stage {index}: {stage.name}] (tentatives: {stage.attempts})"
        )
        status_label = "SUCCESS" if stage.success else "ERROR"
        print(f"  status : {status_label}")
        if stage.justification:
            print(f"  justification : {stage.justification}")
        if stage.payload:
            print("  OUTPUT:")
            pretty = json.dumps(stage.payload, indent=2, ensure_ascii=False)
            print(textwrap.indent(pretty, "    "))

    def _print_scope_decision(self, decision: ScopeDecision) -> None:
        """Dump the :class:`ScopeDecision` structure to stdout when verbose mode is enabled."""
        separator = "=" * 72
        print(f"\n{separator}\nScope Decision\n{separator}")
        print(f"Status : {decision.status}")
        if decision.reply:
            print(f"Reply  : {decision.reply}")
        if decision.reason:
            print(f"Reason : {decision.reason}")

    def _print_preprocess_stages(self, stages: List[StageExecution]) -> None:
        """Render every recorded preprocess stage to help debug clarifier/decomposer/router."""
        if not stages:
            return
        separator = "=" * 72
        print(f"\n{separator}\nQuestion Preprocess\n{separator}")
        for idx, stage in enumerate(stages, start=1):
            self._print_stage_block(stage, idx)

    def _print_llm_registry(self) -> None:
        """Display provider/model pairs for every LLM-backed component when verbose."""

        if not self.llm_registry:
            return
        separator = "=" * 72
        print(f"\n{separator}\nLLM Components\n{separator}")
        for name, pair in sorted(self.llm_registry.items()):
            print(f"{name:<20} -> {pair}")

    def _print_thought_graph_report(self, report: Optional[Dict[str, Any]]) -> None:
        """Print the chronological list of thought steps plus summary statistics."""
        if not report:
            return
        separator = "=" * 72
        
        # Determine mode based on working_memory presence
        working_memory = report.get("working_memory")
        header = "Thinking Loop Summary" if working_memory else "Thought Graph"
        
        print(f"\n{separator}\n{header}\n{separator}")
        steps = report.get("steps") or []
        
        # In thinking loop mode, nodes were already displayed during execution
        # So we only show the summary here
        if working_memory:
            if not steps:
                print("No actions executed.")
            else:
                print(f"Actions executed: {len(steps)}")
            self._print_thought_graph_summary(report, steps)
            return
        
        # Legacy thought graph mode: show full node walkthrough
        if not steps:
            print("No actions executed.")
            self._print_thought_graph_summary(report, steps)
            return

        print("Node Walkthrough:")
        for idx, step in enumerate(steps, start=1):
            self._print_thought_step_summary(step, idx)

        self._print_thought_graph_summary(report, steps)

    def _print_thought_tree(self, report: Optional[Dict[str, Any]]) -> None:
        """Display the ASCII tree for visual debugging.
        
        Uses pre-rendered tree from WorkingMemory if available (thinking loop),
        otherwise builds from steps (legacy thought graph).
        """
        if not report:
            return
        
        separator = "=" * 72
        print(f"\n{separator}\nThought Tree\n{separator}")
        
        # Prefer pre-rendered tree from thinking loop (has full summaries)
        thought_tree = report.get("thought_tree")
        if thought_tree:
            print(thought_tree)
            return
        
        # Fallback to building tree from steps (legacy mode)
        steps = report.get("steps") or []
        if not steps:
            print("(empty)")
            return
        tree_ascii = render_ascii_tree(steps, root_label="root")
        print(tree_ascii)

    def _print_thought_step_summary(self, step: Dict[str, Any], idx: int) -> None:
        """Inspect a single serialized node and expose key telemetry fields."""
        separator = "-" * 56
        node_id = step.get("display_id") or step.get("node_id", "?")
        action = step.get("action", "?")
        depth = step.get("depth", 0)
        
        # Visual depth indicator
        depth_prefix = "│  " * (depth if isinstance(depth, int) else 0)
        
        print(f"\n{separator}")
        header = f"[Node {node_id}: {action}] (depth={depth}, tentatives: {step.get('attempts', 1)})"
        print(header)
        print(f"{depth_prefix}  outcome : {'SUCCESS' if step.get('success') else 'INFO'}")
        justification = step.get("justification")
        if justification:
            print(f"{depth_prefix}  justification : {justification}")
        summary = step.get("summary")
        if summary:
            print(f"{depth_prefix}  note   : {summary}")
        output = step.get("output") or {}
        if output:
            self._print_node_output(output, depth_prefix, action_type=action)

    def _print_node_output(self, output: Dict[str, Any], depth_prefix: str = "", action_type: str = "") -> None:
        """Pretty-print semantic hits or syntax commands embedded in a node output.
        
        Uses head/tail display for long outputs based on verbose_limits config.
        """
        print(f"{depth_prefix}  OUTPUT DETAILS:")
        handled: Set[str] = set()
        
        # Get max lines for this action type
        max_lines = self.verbose_limits.get(action_type, self.verbose_limits.get("default", 20))
        
        question = output.get("question")
        if question:
            print(f"{depth_prefix}    question : {question}")
            handled.add("question")

        commands = output.get("commands") or []
        if commands:
            handled.add("commands")
            for idx, command in enumerate(commands, start=1):
                print(f"    Command #{idx}:")
                print(f"      cmd       : {command.get('command', '-')}")
                if command.get("justification"):
                    print(f"      reason    : {command['justification']}")
                print(f"      status    : {command.get('returncode', '-')}")
                stdout = command.get("stdout", "").rstrip() or "<empty>"
                print("      stdout    :")
                # Apply head/tail truncation
                stdout_truncated = self._truncate_text_headtail(stdout, max_lines)
                print(textwrap.indent(stdout_truncated, "        "))
                stderr = command.get("stderr")
                if stderr:
                    print("      stderr    :")
                    print(textwrap.indent(stderr.rstrip(), "        "))

        hits = output.get("hits") or []
        if hits:
            handled.add("hits")
            for idx, hit in enumerate(hits, start=1):
                print(f"    Hit #{idx}:")
                script_id = hit.get("script_id", "-")
                path = hit.get("file_path", "-")
                line_start = hit.get("line_start", "?")
                line_end = hit.get("line_end", "?")
                summary = hit.get("summary") or (hit.get("text") or "").strip()
                if summary and len(summary) > 280:
                    summary = summary[:277] + "..."
                print(f"      script    : {script_id}")
                print(f"      path      : {path}")
                print(f"      lines     : {line_start}-{line_end}")
                score = hit.get("score")
                if score is not None:
                    try:
                        print(f"      score     : {float(score):.3f}")
                    except (TypeError, ValueError):
                        print(f"      score     : {score}")
                if summary:
                    print(f"      summary   : {summary}")

        # For syntax agent: only show summary in "other", not counts/unique_modules
        # (those belong to compute agent analysis)
        if action_type == "syntax":
            # Summary is the main output for syntax
            summary_text = output.get("summary", "")
            if summary_text:
                print(f"    summary   : {summary_text}")
            # Skip internal data in "other" - keep display clean
            handled.update(["summary", "occurrences_data", "occurrences_count", 
                          "scripts_count", "total_matches"])
        
        remaining = {k: v for k, v in output.items() if k not in handled}
        if remaining:
            print("    other:")
            pretty = json.dumps(remaining, ensure_ascii=False, indent=2)
            # Apply head/tail truncation to JSON output too
            pretty_truncated = self._truncate_text_headtail(pretty, max_lines)
            print(textwrap.indent(pretty_truncated, "      "))
    
    def _truncate_text_headtail(self, text: str, max_lines: int) -> str:
        """Truncate text showing first N/2 and last N/2 lines if too long."""
        if not text:
            return text
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
        
        half = max_lines // 2
        head = lines[:half]
        tail = lines[-half:]
        omitted = len(lines) - max_lines
        
        return '\n'.join(head) + f'\n\n... [{omitted} lignes omises] ...\n\n' + '\n'.join(tail)

    def _print_thought_graph_summary(
        self, report: Dict[str, Any], steps: List[Dict[str, Any]]
    ) -> None:
        """Summarize planner health (status, depth, pending actions) after verbose runs."""
        separator = "=" * 72
        
        # Check if this is a thinking-loop report (has working_memory)
        working_memory = report.get("working_memory")
        
        if working_memory:
            # New thinking-first display
            print(f"\n{separator}\nThinking Loop Stats\n{separator}")
            status = report.get("status", "unknown")
            iterations = report.get("iterations", "-")
            confidence = report.get("final_confidence", 0.0)
            total_nodes = len(steps)
            success_nodes = sum(1 for step in steps if step.get("success"))
            
            print(f"Status       : {status}")
            print(f"Iterations   : {iterations}")
            print(f"Confidence   : {confidence:.2f}" if isinstance(confidence, (int, float)) else f"Confidence   : {confidence}")
            print(f"Actions      : {success_nodes}/{total_nodes} successful")
            
            # Hypotheses
            hypotheses = working_memory.get("hypotheses", [])
            if hypotheses:
                print(f"\nHypotheses ({len(hypotheses)}):")
                for h in hypotheses:  # Show ALL hypotheses in verbose mode
                    claim = h.get("claim", "")
                    conf = h.get("confidence", 0)
                    print(f"  • [{conf:.2f}] {claim}")
            
            # Patterns discovered
            patterns = working_memory.get("patterns", [])
            if patterns:
                print(f"\nPatterns discovered ({len(patterns)}):")
                for p in list(patterns):  # Show ALL patterns in verbose mode
                    print(f"  • {p}")
            
            # Scripts of interest
            scripts = working_memory.get("scripts_of_interest", [])
            if scripts:
                print(f"\nScripts identified ({len(scripts)}):")
                for s in scripts:  # Show ALL scripts in verbose mode
                    script_id = s.get("script", s[0] if isinstance(s, (list, tuple)) else s)
                    reason = s.get("reason", s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else "")
                    if reason:
                        print(f"  • {script_id} - {reason}")
                    else:
                        print(f"  • {script_id}")
            
            # Open questions remaining
            open_q = working_memory.get("open_questions", [])
            if open_q:
                print(f"\nOpen questions ({len(open_q)}):")
                for q in open_q:  # Show ALL questions in verbose mode
                    print(f"  ? {q}")
            
            # Dead ends
            dead_ends = working_memory.get("dead_ends", [])
            if dead_ends:
                print(f"\nDead ends ({len(dead_ends)}):")
                for de in dead_ends[:2]:
                    path = de.get("path", de[0] if isinstance(de, (list, tuple)) else de)
                    print(f"  ✗ {path}")
        else:
            # Legacy thought graph display
            print(f"\n{separator}\nThought Graph Stats\n{separator}")
            status = report.get("status", "unknown")
            decision = report.get("decision", "-")
            iterations = report.get("iterations", "-")
            reason = report.get("reason", "-")
            total_nodes = len(steps)
            success_nodes = sum(1 for step in steps if step.get("success"))
            max_depth = max((step.get("depth", 0) for step in steps), default=0)
            print(f"Status       : {status}")
            print(f"Decision     : {decision}")
            print(f"Iterations   : {iterations}")
            print(f"Reason       : {reason}")
            print(f"Nodes        : {success_nodes}/{total_nodes} successful")
            print(f"Max depth    : {max_depth}")
            pending = report.get("pending_router_actions") or []
            if pending:
                print(f"Pending router actions : {', '.join(pending)}")

    def _print_final_result_stats(
        self,
        question: str,
        decision: ScopeDecision,
        preprocess: PreprocessBundle,
        thought_report: Dict[str, Any],
        final_payload: Dict[str, Any],
    ) -> None:
        """Show the stitched response along with preprocess/graph checkpoints."""
        summary = self.build_run_summary(
            question=question,
            decision=decision,
            preprocess=preprocess,
            thought_report=thought_report,
            payload=final_payload,
        )
        separator = "=" * 72
        
        # Header
        print(f"\n{separator}")
        print("SYNTHESIZER - FINAL RESULT")
        print(separator)
        
        # Question reminder
        print(f"Question     : {question}")
        
        # Agents used summary (compact: "agent *N" format)
        agents = summary.get("agents", {})
        if agents:
            agent_parts = [f"{k} *{v}" for k, v in agents.items() if v > 0]
            if agent_parts:
                print(f"Agents       : {', '.join(agent_parts)}")
        
        payload = summary.get("payload", {})
        
        # Termination status
        term_status = payload.get("termination_status", "completed")
        limits_hit = payload.get("limits_hit", [])
        is_complete = payload.get("is_complete", True)
        
        # Show status with appropriate emoji/indicator
        if term_status == "completed":
            status_display = "✓ Terminé (confiance atteinte)"
        elif term_status == "limit_reached":
            status_display = f"⚠ Limite(s) atteinte(s): {', '.join(limits_hit)}"
        elif term_status == "forced_stop":
            status_display = "✗ Arrêt forcé"
        else:
            status_display = term_status
        print(f"Status       : {status_display}")
        
        # Confidence if available
        confidence = payload.get("answer_confidence")
        if confidence is not None:
            conf_indicator = "✓" if confidence >= 0.7 else ("~" if confidence >= 0.4 else "?")
            print(f"Confidence   : {confidence:.2f} {conf_indicator}")
        
        # Completeness indicator
        if not is_complete:
            print(f"Complet      : Non - résultats potentiellement partiels")
            needs_more = payload.get("needs_more_investigation")
            if needs_more:
                print(f"À creuser    : {needs_more}")
        
        # References count
        refs = payload.get("references", [])
        if refs:
            print(f"References   : {len(refs)} script(s)")
        
        # Answer
        print(f"\n{'─'*40}")
        print("ANSWER:")
        print(f"{'─'*40}")
        answer = payload.get("answer", "")
        if answer:
            # Wrap answer text nicely
            import textwrap as tw
            wrapped = tw.fill(answer, width=70)
            print(wrapped)
        else:
            print("(No answer generated)")
        
        # Highlights if any
        highlights = payload.get("answer_highlights", [])
        if highlights:
            print(f"\n{'─'*40}")
            print("HIGHLIGHTS:")
            for h in highlights:
                print(f"  • {h}")
        
        # Appendix - ALWAYS show complete (never truncate)
        appendix = payload.get("appendix", [])
        if appendix:
            print(f"\n{'─'*40}")
            print(f"APPENDIX ({len(appendix)} entries):")
            for entry in appendix:
                print(f"  {entry}")
        
        print(separator)

    def _build_structured_response(
        self,
        *,
        question: str,
        decision: ScopeDecision,
        preprocess: PreprocessBundle,
        thought_report: Optional[Dict[str, Any]],
        context: str,
        working_memory: Optional[WorkingMemory] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
        """Assemble the final user payload plus evidence/references/agent stats."""
        references_payload = self._extract_references(thought_report)
        
        # Enrich references with working memory insights if available
        if working_memory:
            # Add scripts_of_interest that might not be in formal references
            for script_id, reason in working_memory.scripts_of_interest:
                script_id_clean = script_id.strip()
                if script_id_clean and script_id_clean not in [r.get("script_id") for r in references_payload]:
                    display = self.script_mapping.get(script_id_clean, script_id_clean)
                    references_payload.append({
                        "script_id": script_id_clean,
                        "display_name": display,
                        "sources": ["working_memory"],
                        "evidence": [f"Identified as relevant: {reason}"],
                    })
        
        simple_references = [
            ref.get("display_name") or ref.get("script_id")
            for ref in references_payload
            if ref.get("display_name") or ref.get("script_id")
        ]
        agent_stats = self._count_agent_calls(thought_report)
        thought_summary = self._summarize_thought_report(thought_report)
        router_intent = (preprocess.router or {}).get("intent")
        
        # Extract termination info from thought_report
        termination_info = self._extract_termination_info(thought_report)
        
        # CRITICAL: Enrich context with key findings from thinking loop
        # so the Answer Synthesizer has the extracted data, not just raw context
        enriched_context = self._enrich_context_with_findings(context, thought_report, working_memory)
        
        answer, appendix, answer_meta = self._compose_final_answer(
            question=question,
            router_intent=router_intent,
            references=references_payload,
            agent_stats=agent_stats,
            context=enriched_context,
            termination_info=termination_info,
        )
        payload = {
            "status": thought_summary.get("status") or "completed",
            "answer": answer,
            "appendix": appendix,
            "references": simple_references,
            "agents": agent_stats,
        }
        # Include termination info in payload
        if termination_info:
            payload["termination_status"] = termination_info.get("status", "completed")
            if termination_info.get("limits_hit"):
                payload["limits_hit"] = termination_info["limits_hit"]
        
        confidence = answer_meta.get("confidence") if answer_meta else None
        if confidence is not None:
            payload["answer_confidence"] = confidence
        
        # Include completeness info from synthesizer
        is_complete = answer_meta.get("is_complete", True) if answer_meta else True
        payload["is_complete"] = is_complete
        needs_more = answer_meta.get("needs_more_investigation") if answer_meta else None
        if needs_more:
            payload["needs_more_investigation"] = needs_more
        
        highlights = (answer_meta or {}).get("highlights") or []
        if highlights:
            payload["answer_highlights"] = highlights
        return payload, references_payload, agent_stats

    def _summarize_thought_report(
        self, report: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract a light summary of planner steps for logging and responses."""
        if not report:
            return {"status": "skipped", "steps": []}
        steps = report.get("steps") or []
        summarized_steps: List[Dict[str, Any]] = []
        for step in steps:
            summary_entry = {
                "node_id": step.get("node_id"),
                "parent": step.get("parent_id"),
                "action": step.get("action"),
                "success": step.get("success"),
                "summary": step.get("summary"),
                "justification": step.get("justification"),
                "attempts": step.get("attempts"),
                "display_id": step.get("display_id"),
            }
            output = step.get("output") or {}
            commands = output.get("commands") or []
            if commands:
                summary_entry["commands"] = [cmd.get("command") for cmd in commands]
            hits = output.get("hits") or []
            if hits:
                summary_entry["hits"] = [
                    {
                        "script_id": hit.get("script_id"),
                        "path": hit.get("file_path"),
                        "score": hit.get("score"),
                    }
                    for hit in hits[:3]
                ]
            summarized_steps.append(summary_entry)
        return {
            "status": report.get("status", "unknown"),
            "decision": report.get("decision"),
            "iterations": report.get("iterations"),
            "reason": report.get("reason"),
            "steps": summarized_steps,
        }

    def _extract_termination_info(
        self, thought_report: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract termination status and reason from the thinking loop report.
        
        Returns a dict with:
        - status: "completed" | "limit_reached" | "forced_stop" | "skipped"
        - reason: human-readable explanation
        - confidence: final confidence from the thinking loop
        - limits_hit: list of limits that were reached (if any)
        """
        if not thought_report:
            return {
                "status": "skipped",
                "reason": "No thinking loop executed",
                "confidence": 0.0,
                "limits_hit": [],
            }
        
        status = thought_report.get("status", "unknown")
        forced_stop_reason = thought_report.get("forced_stop_reason")
        fail_reason = thought_report.get("fail_reason")
        confidence = thought_report.get("final_confidence", 0.0)
        
        # Check for limits
        stats = thought_report.get("stats", {})
        limits = thought_report.get("limits", {})
        limits_hit: List[str] = []
        
        if stats.get("iterations", 0) >= limits.get("max_iterations", 999):
            limits_hit.append(f"max_iterations ({limits.get('max_iterations')})")
        if stats.get("total_steps", 0) >= limits.get("max_total_steps", 999):
            limits_hit.append(f"max_total_steps ({limits.get('max_total_steps')})")
        if stats.get("max_depth_reached", 0) >= limits.get("max_depth", 999):
            limits_hit.append(f"max_depth ({limits.get('max_depth')})")
        
        # Determine effective status
        if forced_stop_reason:
            effective_status = "forced_stop"
            reason = forced_stop_reason
        elif limits_hit:
            effective_status = "limit_reached"
            reason = f"Limites atteintes: {', '.join(limits_hit)}"
        elif fail_reason:
            effective_status = "failed"
            reason = fail_reason
        elif status == "completed":
            effective_status = "completed"
            reason = "Confiance suffisante atteinte"
        else:
            effective_status = status
            reason = "Processus terminé"
        
        return {
            "status": effective_status,
            "reason": reason,
            "confidence": confidence,
            "limits_hit": limits_hit,
        }

    def _enrich_context_with_findings(
        self,
        original_context: str,
        thought_report: Optional[Dict[str, Any]],
        working_memory: Optional[WorkingMemory] = None,
    ) -> str:
        """Enrich the original context with key findings from the thinking loop.
        
        This ensures the Answer Synthesizer has access to:
        - unique_modules extracted from syntax searches
        - computed results from the compute agent
        - patterns/hypotheses from working memory
        """
        findings_parts: List[str] = []
        
        if thought_report:
            steps = thought_report.get("steps") or []
            for step in steps:
                output = step.get("output") or {}
                action = step.get("action", "")
                
                # Extract computed results from compute agent (THE source of truth for counts/lists)
                computed_result = output.get("result")
                if computed_result is not None and action == "compute":
                    if isinstance(computed_result, list):
                        result_str = "\n".join(f"  - {r}" for r in computed_result)
                        findings_parts.append(
                            f"[RÉSULTAT CALCULÉ PAR COMPUTE - {len(computed_result)} élément(s)]\n{result_str}"
                        )
                    elif isinstance(computed_result, dict):
                        result_str = "\n".join(f"  - {k}: {v}" for k, v in computed_result.items())
                        findings_parts.append(f"[RÉSULTAT CALCULÉ PAR COMPUTE]\n{result_str}")
                    else:
                        findings_parts.append(f"[RÉSULTAT CALCULÉ PAR COMPUTE]\n  {computed_result}")
                
                # For syntax: just note how many occurrences were found (raw data)
                if action == "syntax":
                    occurrences_count = output.get("occurrences_count", 0)
                    scripts_count = output.get("scripts_count", 0)
                    if occurrences_count > 0:
                        findings_parts.append(
                            f"[DONNÉES BRUTES SYNTAX] {occurrences_count} occurrence(s) dans {scripts_count} script(s)"
                        )
        
        # Add working memory patterns if available
        if working_memory:
            if working_memory.patterns:
                patterns_list = list(working_memory.patterns)[:5]
                patterns_str = "\n".join(f"  - {p}" for p in patterns_list)
                findings_parts.append(f"[PATTERNS IDENTIFIÉS]\n{patterns_str}")
            
            if working_memory.hypotheses:
                hyp_str = "\n".join(f"  - [{h.confidence:.2f}] {h.claim}" for h in working_memory.hypotheses[:3])
                findings_parts.append(f"[HYPOTHÈSES]\n{hyp_str}")
        
        if not findings_parts:
            return original_context
        
        # Combine original context with findings
        findings_section = "\n\n".join(findings_parts)
        enriched = f"""{original_context}

--- RÉSULTATS DE L'ANALYSE DU THINKING LOOP ---
{findings_section}
--- FIN DES RÉSULTATS ---"""
        
        return enriched

    def _extract_references(
        self, report: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect semantic hits and syntax excerpts into a deduplicated evidence list."""
        if not report:
            return []
        steps = report.get("steps") or []
        references: Dict[str, Dict[str, Any]] = {}
        for step in steps:
            action = step.get("action", "")
            output = step.get("output") or {}
            # Semantic hits
            for hit in output.get("hits") or []:
                script_id = (hit.get("script_id") or "").strip() or None
                display = self.script_mapping.get(script_id, hit.get("file_path") or script_id or "")
                key = script_id or display
                if not key:
                    continue
                entry = references.setdefault(
                    key,
                    {
                        "script_id": script_id,
                        "display_name": display,
                        "sources": set(),
                        "evidence": [],
                    },
                )
                entry["sources"].add(action or "semantic")
                evidence = {
                    "type": "semantic",
                    "score": hit.get("score"),
                    "summary": (hit.get("summary") or (hit.get("text") or "").strip())[:280],
                    "lines": [hit.get("line_start"), hit.get("line_end")],
                }
                entry["evidence"].append(evidence)
            # Syntax outputs
            for command in output.get("commands") or []:
                stdout = command.get("stdout") or ""
                parsed = self._maybe_parse_json(stdout)
                if isinstance(parsed, dict) and parsed.get("results"):
                    for result in parsed.get("results", []):
                        script_id = (str(result.get("script_id")) or "").strip() or None
                        path = result.get("file_path") or ""
                        display = self.script_mapping.get(script_id, path or script_id or "")
                        key = script_id or display
                        if not key:
                            continue
                        entry = references.setdefault(
                            key,
                            {
                                "script_id": script_id,
                                "display_name": display,
                                "sources": set(),
                                "evidence": [],
                            },
                        )
                        entry["sources"].add(action or "syntax")
                        for hit in result.get("hits", []) or [{}]:
                            evidence = {
                                "type": "syntax",
                                "line": hit.get("line"),
                                "excerpt": (hit.get("raw") or hit.get("resolved_path") or "").strip(),
                                "verb": hit.get("verb"),
                                "path": hit.get("resolved_path") or path,
                            }
                            entry["evidence"].append(evidence)
                else:
                    for line in stdout.splitlines():
                        path, line_no, excerpt = self._parse_stdout_line(line)
                        if not path:
                            continue
                        script_id = self._path_to_id(path) or self._infer_id_from_path(path)
                        display = self.script_mapping.get(script_id, path)
                        key = script_id or display
                        entry = references.setdefault(
                            key,
                            {
                                "script_id": script_id,
                                "display_name": display,
                                "sources": set(),
                                "evidence": [],
                            },
                        )
                        entry["sources"].add(action or "syntax")
                        evidence = {
                            "type": "syntax",
                            "line": line_no,
                            "excerpt": excerpt.strip(),
                        }
                        entry["evidence"].append(evidence)
        normalized: List[Dict[str, Any]] = []
        for entry in references.values():
            sources = sorted(entry["sources"])
            if "syntax" not in sources:
                continue
            evidence = entry["evidence"][:5]
            normalized.append(
                {
                    "script_id": entry["script_id"],
                    "display_name": entry["display_name"],
                    "sources": sources,
                    "evidence": evidence,
                }
            )
        normalized.sort(key=lambda item: item["display_name"] or item.get("script_id") or "")
        return normalized

    def build_run_summary(
        self,
        *,
        question: str,
        decision: ScopeDecision,
        preprocess: Optional[PreprocessBundle],
        thought_report: Optional[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate human-readable telemetry for CLI/benchmark consumers."""

        bundle = preprocess or PreprocessBundle(normalized_question=question)
        stage_lines = [f"{stage.name}={'OK' if stage.success else 'FAIL'}" for stage in bundle.stages]
        stage_details = [
            {
                "name": stage.name,
                "success": stage.success,
                "attempts": stage.attempts,
                "justification": stage.justification,
            }
            for stage in bundle.stages
        ]
        stage_status = " | ".join(stage_lines) if stage_lines else "n/a"
        thought = thought_report or {}
        steps = thought.get("steps") or []
        working_memory = thought.get("working_memory")
        payload = payload or {}
        
        # Determine if this was a thinking-loop run
        is_thinking_loop = working_memory is not None
        
        summary = {
            "process_flow": "Scope → Preprocess → Thinking Loop → Synthesis" if is_thinking_loop else "Scope → Preprocess → Thought Graph → Synthesis",
            "question": question,
            "scope_reply": decision.reply,
            "preprocess_status": stage_status,
            "preprocess_details": stage_details,
            "agents": payload.get("agents", {}),
            "answer_preview": payload.get("answer") or payload.get("message") or "",
            "payload": payload,
        }
        
        if is_thinking_loop:
            stats = thought.get("stats", {})
            limits = thought.get("limits", {})
            summary["thinking_loop"] = {
                "status": thought.get("status", "n/a"),
                "iterations": f"{stats.get('iterations', 0)}/{limits.get('max_iterations', '?')}",
                "total_steps": f"{stats.get('total_steps', 0)}/{limits.get('max_total_steps', '?')}",
                "max_depth_reached": f"{stats.get('max_depth_reached', 0)}/{limits.get('max_depth', '?')}",
                "retries": stats.get("retries", 0),
                "splits_created": stats.get("splits_created", 0),
                "branches_explored": stats.get("branches_explored", 0),
                "confidence": thought.get("final_confidence", 0.0),
                "actions": len(steps),
                "hypotheses": len(working_memory.get("hypotheses", [])),
                "patterns": len(working_memory.get("patterns", [])),
                "scripts_found": len(working_memory.get("scripts_of_interest", [])),
                "nodes_count": thought.get("nodes_count", 0),
            }
            if thought.get("forced_stop_reason"):
                summary["thinking_loop"]["forced_stop_reason"] = thought["forced_stop_reason"]
            if thought.get("thought_tree"):
                summary["thought_tree"] = thought["thought_tree"]
        else:
            summary["thought_graph"] = {
                "status": thought.get("status", "n/a"),
                "nodes": len(steps),
                "decision": thought.get("decision", "-"),
            }
        
        return summary

    def summarize_outcome(self, *, question: str, outcome: RunOutcome) -> Dict[str, Any]:
        """Public helper exposing :meth:`build_run_summary` for external callers."""

        return self.build_run_summary(
            question=question,
            decision=outcome.decision,
            preprocess=outcome.preprocess,
            thought_report=outcome.thought_report,
            payload=outcome.payload,
        )

    def format_run_summary(self, summary: Dict[str, Any]) -> str:
        """Return a formatted multiline string highlighting key run stats."""

        lines = [
            f"Process Flow : {summary.get('process_flow', '')}",
            f"Question     : {summary.get('question', '')}",
            f"Scope reply  : {summary.get('scope_reply', '')}",
            f"Preprocess   : {summary.get('preprocess_status', 'n/a')}",
        ]
        
        # Check if thinking loop or thought graph
        thinking = summary.get("thinking_loop")
        thought = summary.get("thought_graph")
        
        if thinking:
            confidence = thinking.get('confidence', 0)
            conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
            lines.append(
                f"Thinking Loop: status={thinking.get('status', 'n/a')} "
                f"iterations={thinking.get('iterations', '?')} "
                f"steps={thinking.get('total_steps', '?')} "
                f"max_depth={thinking.get('max_depth_reached', '?')}"
            )
            lines.append(
                f"             : confidence={conf_str} "
                f"hypotheses={thinking.get('hypotheses', 0)} "
                f"patterns={thinking.get('patterns', 0)} "
                f"nodes={thinking.get('nodes_count', 0)}"
            )
            if thinking.get("splits_created", 0) > 0:
                lines.append(
                    f"             : splits={thinking.get('splits_created', 0)} "
                    f"branches_explored={thinking.get('branches_explored', 0)} "
                    f"retries={thinking.get('retries', 0)}"
                )
            if thinking.get("forced_stop_reason"):
                lines.append(f"             : FORCED STOP: {thinking['forced_stop_reason']}")
        elif thought:
            lines.append(
                f"Thought Graph: status={thought.get('status', 'n/a')} "
                f"nodes={thought.get('nodes', 0)} "
                f"decision={thought.get('decision', '-')}"
            )
        
        lines.append(f"Agents       : {summary.get('agents', {})}")
        
        preview = summary.get("answer_preview")
        if preview:
            lines.append(f"Answer preview: {preview}")
        return "\n".join(lines)

    @staticmethod
    def _maybe_parse_json(payload: str) -> Optional[Any]:
        """Best-effort JSON parser used on command stdout payloads."""

        text = (payload or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _parse_stdout_line(self, line: str) -> Tuple[str, Optional[str], str]:
        """Parse ``path:line:excerpt`` or ``path-line-excerpt`` formats emitted by SyntaxAgent."""
        if not line or line.startswith("--"):
            return ("", None, "")
        colon_parts = line.split(":", 2)
        if len(colon_parts) >= 2 and colon_parts[1].isdigit():
            rest = colon_parts[2] if len(colon_parts) == 3 else ""
            return (colon_parts[0], colon_parts[1], rest)
        match = re.match(r"^(?P<path>.+?)-( ?)?(?P<line>\d+)-( ?)?(?P<rest>.*)$", line)
        if match:
            return (match.group("path"), match.group("line"), match.group("rest"))
        return ("", None, line)

    def _path_to_id(self, path: str) -> Optional[str]:
        """Return the NVN identifier stored during :meth:`_build_path_index` lookup."""
        key = self._normalize_path_key(path)
        return self.path_to_id.get(key)

    def _infer_id_from_path(self, path: str) -> Optional[str]:
        """Heuristic to recover script IDs when a path was not part of the lookup table."""
        match = re.search(r"(\d{5})\.nvn", path)
        if match:
            return match.group(1)
        match = re.search(r"/(\d{5})(?:\D|$)", path)
        if match:
            return match.group(1)
        return None

    def _count_agent_calls(self, report: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Count how many semantic/syntax nodes were executed for telemetry."""
        stats = {"syntax": 0, "semantic": 0}
        if not report:
            return stats
        for step in report.get("steps") or []:
            action = step.get("action")
            if action in stats:
                stats[action] += 1
        return stats

    def _compose_final_answer(
        self,
        *,
        question: str,
        router_intent: Optional[str],
        references: List[Dict[str, Any]],
        agent_stats: Dict[str, int],
        context: str,
        termination_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Use the answer LLM (with fallback) to narrate findings + appendix."""

        intent = router_intent or self._infer_intent_from_agents(agent_stats)
        normalized_context = (context or "").strip()
        references = references or []
        formatted_refs = [self._format_reference_entry(ref) for ref in references]
        inline_limit = max(1, self.synthesizer_inline_limit)
        inline_refs = references[:inline_limit]
        inline_payload = [self._prepare_answer_evidence_entry(ref) for ref in inline_refs]
        appendix_required = len(references) > inline_limit
        appendix = [f"- {entry}" for entry in formatted_refs] if appendix_required else []
        fallback_answer = self._build_fallback_answer_text(
            question=question,
            intent=intent,
            formatted_refs=formatted_refs,
            reference_count=len(references),
        )
        answer_text = fallback_answer
        meta: Dict[str, Any] = {"confidence": None, "highlights": [], "is_complete": True}
        appendix_flag = "appendix" if appendix_required else "inline"
        overflow_count = max(0, len(references) - len(inline_refs))
        
        # Extract termination info with defaults
        term_info = termination_info or {"status": "completed", "reason": "", "confidence": 0.0}
        termination_status = term_info.get("status", "completed")
        termination_reason = term_info.get("reason", "")
        thinking_confidence = term_info.get("confidence", 0.0)

        if self.synthesizer:
            prompt = self.synthesizer_prompt_template.format(
                question=question.strip(),
                intent=intent,
                context=normalized_context or "Contexte non fourni.",
                inline_evidence=json.dumps(inline_payload, ensure_ascii=False),
                agent_stats=json.dumps(agent_stats, ensure_ascii=False),
                appendix_flag=appendix_flag,
                max_inline_refs=inline_limit,
                evidence_count=len(references),
                overflow_count=overflow_count,
                termination_status=termination_status,
                termination_reason=termination_reason,
                thinking_confidence=thinking_confidence,
            )
            synth_payload = self._invoke_synthesizer(prompt)
            if synth_payload:
                llm_answer = (synth_payload.get("answer") or "").strip()
                if llm_answer:
                    answer_text = llm_answer
                confidence = self._normalize_confidence(synth_payload.get("confidence"))
                if confidence is not None:
                    meta["confidence"] = confidence
                # New fields for completeness tracking
                meta["is_complete"] = synth_payload.get("is_complete", True)
                needs_more = synth_payload.get("needs_more_investigation")
                if needs_more:
                    meta["needs_more_investigation"] = needs_more
                highlights_raw = synth_payload.get("highlights") or []
                if isinstance(highlights_raw, list):
                    filtered = [str(item).strip() for item in highlights_raw if str(item).strip()]
                    if filtered:
                        meta["highlights"] = filtered
                elif isinstance(highlights_raw, str) and highlights_raw.strip():
                    meta["highlights"] = [highlights_raw.strip()]

        return (answer_text, appendix, meta)

    def _prepare_answer_evidence_entry(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        """Trim evidence dicts so answer prompts stay compact."""

        label = ref.get("display_name") or ref.get("script_id") or "(script inconnu)"
        evidence = ref.get("evidence") or []
        primary = evidence[0] if evidence else {}
        
        # Handle both dict and string evidence types
        if isinstance(primary, dict):
            snippet = (primary.get("excerpt") or primary.get("summary") or "").strip()
            lines = primary.get("lines") or primary.get("line")
        elif isinstance(primary, str):
            snippet = primary.strip()
            lines = None
        else:
            snippet = ""
            lines = None
            
        entry: Dict[str, Any] = {
            "label": label,
            "script_id": ref.get("script_id"),
            "sources": ref.get("sources"),
            "snippet": snippet,
        }
        if lines:
            entry["lines"] = lines
        return entry

    def _invoke_synthesizer(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the synthesizer LLM and parse JSON, swallowing failures gracefully."""

        if not self.synthesizer:
            return None
        raw = ""
        try:
            raw = self.synthesizer.generate_response(prompt=prompt)
            parsed = parse_llm_json(raw)
        except Exception as exc:  # pragma: no cover - network/runtime
            if self.verbose:
                print(f"[warn] Synthesizer call failed: {exc}")
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _normalize_confidence(value: Any) -> Optional[float]:
        """Clamp LLM-provided confidence to [0, 1] or return ``None``."""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, confidence))

    def _build_fallback_answer_text(
        self,
        *,
        question: str,
        intent: str,
        formatted_refs: List[str],
        reference_count: int,
    ) -> str:
        """Fallback narrative used whenever the answer LLM is unavailable."""

        if reference_count <= 0:
            suggestion = (
                "Aucune évidence exploitable n'a été trouvée dans la profondeur allouée."
                " Relancer avec un autre chemin ou augmenter le périmètre pourrait aider."
            )
            return f"Question: {question}\n{suggestion}"

        if intent == "syntax":
            base = "Analyse déterministe"
        elif intent == "semantic":
            base = "Analyse contextuelle"
        else:
            base = "Analyse hybride"

        count_phrase = self._describe_reference_count(reference_count)
        verb = "répond" if reference_count == 1 else "répondent"
        prelude = f"{base} : {count_phrase} {verb} à la question."

        if reference_count <= self.synthesizer_inline_limit:
            details = ", ".join(formatted_refs)
            return f"{prelude} Détail : {details}" if details else prelude

        return f"{prelude} Voir l'appendice pour le détail des scripts et extraits."

    def _infer_intent_from_agents(self, stats: Dict[str, int]) -> str:
        """Translate agent usage counts into a high-level intent label."""
        syntax_calls = stats.get("syntax", 0)
        semantic_calls = stats.get("semantic", 0)
        if syntax_calls and semantic_calls:
            return "hybrid"
        if syntax_calls:
            return "syntax"
        if semantic_calls:
            return "semantic"
        return "unknown"

    def _format_reference_entry(self, ref: Dict[str, Any]) -> str:
        """Collapse a reference dict into a single line for appendices or inline summaries."""
        label = ref.get("display_name") or ref.get("script_id") or "(script inconnu)"
        evidence = ref.get("evidence") or []
        snippet = ""
        if evidence:
            first = evidence[0]
            # Handle both dict evidence (from agents) and string evidence (from working_memory)
            if isinstance(first, dict):
                snippet = first.get("excerpt") or first.get("summary") or first.get("raw") or ""
            elif isinstance(first, str):
                snippet = first
            else:
                snippet = str(first) if first else ""
            snippet = snippet.strip()
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
        if snippet:
            return f"{label} — {snippet}"
        return label

    def _describe_reference_count(self, count: int) -> str:
        """Return a French human-readable count label used in the final summary."""
        if count <= 0:
            return "aucun script"
        noun = "script" if count == 1 else "scripts"
        return f"{count} {noun}"

    def _emit_response(
        self,
        *,
        payload: Dict[str, Any],
        decision: ScopeDecision,
        question: str,
        context: str,
        preprocess: Optional[PreprocessBundle],
        thought_report: Optional[Dict[str, Any]],
        stats: Optional[Dict[str, Any]] = None,
        loggable: bool = True,
        as_text: bool = True,
        record_history: bool = True,
    ) -> RunOutcome:
        """Serialize the final payload, persist it if requested, and expose the artifacts."""
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if loggable and record_history:
            self._record_history(
                question=question,
                context=context,
                decision=decision,
                preprocess=preprocess,
                thought_report=thought_report,
                response=payload,
                response_text=response_text,
                stats=stats,
            )
        return RunOutcome(
            payload=payload,
            text=response_text if as_text else response_text,
            decision=decision,
            preprocess=preprocess,
            thought_report=thought_report,
            stats=stats,
        )

    def _build_log_stats(
        self,
        *,
        agent_stats: Dict[str, int],
        references: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the compact telemetry dict stored alongside each JSONL log entry."""
        reference_names = [
            ref.get("display_name") or ref.get("script_id")
            for ref in references
            if ref.get("display_name") or ref.get("script_id")
        ]
        answer_preview = payload.get("answer") or payload.get("message") or ""
        return {
            "status": payload.get("status"),
            "agent_calls": agent_stats,
            "reference_count": len(references),
            "references": reference_names,
            "answer_preview": answer_preview[:280],
        }

    def _record_history(
        self,
        *,
        question: str,
        context: str,
        decision: ScopeDecision,
        preprocess: Optional[PreprocessBundle],
        thought_report: Optional[Dict[str, Any]],
        response: Dict[str, Any],
        response_text: str,
        stats: Optional[Dict[str, Any]],
    ) -> None:
        """Append a JSON line to ``history_file`` summarizing the full run."""
        if not self.history_file:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "context": context,
            "decision": asdict(decision),
            "preprocess": preprocess.serialize() if preprocess else None,
            "thought_report": thought_report,
            "response": response,
            "response_text": response_text,
            "stats": stats or {},
        }
        try:
            with self.history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem issues
            if self.verbose:
                print(f"[warn] Unable to append history log: {exc}")

    def _build_path_index(self, mapping: Dict[str, str]) -> Dict[str, str]:
        """Create a lookup between normalized script paths and their NVN identifiers."""
        index: Dict[str, str] = {}
        for script_id, rel_path in mapping.items():
            if not rel_path:
                continue
            normalized = self._normalize_path_key(rel_path)
            index[normalized] = script_id
            stripped = normalized.lstrip("/")
            index[stripped] = script_id
        return index

    def _normalize_path_key(self, path: str) -> str:
        """Uniformly trim whitespace so lookups succeed regardless of formatting."""
        return (path or "").strip()

    def _prepare_component_llm(
        self,
        component: str,
        cfg: Optional[Dict[str, Any]],
    ) -> LLMClient:
        """Resolve provider/model for a component and instantiate the corresponding client."""
        provider, model = self._resolve_llm_choice(cfg)
        try:
            client = prepare_llm(provider, model=model)
            self.llm_registry[component] = f"{provider}/{model}"
            return client
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize LLM for {component} ({provider}/{model})"
            ) from exc

    def _resolve_llm_choice(
        self, cfg: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Return the provider/model pair, falling back to global defaults when needed."""
        defaults = self.llm_defaults if isinstance(self.llm_defaults, dict) else {}
        provider = self._normalize_llm_value(
            (cfg or {}).get("llm"),
            defaults.get("default_provider", "mistral"),
        )
        model = self._normalize_llm_value(
            (cfg or {}).get("model"),
            defaults.get("default_model", "mistral-large-latest"),
        )
        return provider, model

    @staticmethod
    def _normalize_llm_value(value: Optional[str], fallback: str) -> str:
        if value is None:
            return str(fallback or "mistral")
        text = str(value).strip()
        if not text or text.lower() == "default":
            return str(fallback or "mistral")
        return text

    @staticmethod
    def _build_simple_response(
        status: str, message: str, extras: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Helper to craft minimal JSON payloads for greetings/off-topic/errors."""
        payload = {"status": status, "message": message}
        if extras:
            payload.update(extras)
        return payload

def load_runtime_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration or raise a descriptive error."""

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return load_config(path)


def resolve_history_file(config: Dict[str, Any]) -> Path:
    """Return the history JSONL path (defaults to ``logs/history.jsonl``)."""

    logging_cfg = (
        config.get("main_agent", {}).get("logging", {}) if isinstance(config, dict) else {}
    )
    history_hint = logging_cfg.get("history_file", "logs/oracle-history.jsonl")
    return Path(history_hint).resolve()


def tail_history_file(path: Path, limit: Optional[int], short: bool, index: Optional[int]) -> None:
    """Print history entries in long JSON form or condensed summary."""

    if not path.exists():
        print(f"[tail-logs] History file not found: {path}")
        return
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        print("[tail-logs] History file is empty.")
        return

    def render_entry(raw: str, ordinal: int) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"\n--- Entry {ordinal} (unparsed) ---\n{raw}")
            return
        if short:
            print(format_short_history_entry(payload, ordinal))
        else:
            formatted = json.dumps(payload, ensure_ascii=False, indent=2)
            print(f"\n--- Entry {ordinal} ---\n{formatted}")

    if index is not None:
        if index <= 0 or index > len(lines):
            print(
                f"[tail-logs] Invalid index {index}. File currently contains {len(lines)} entr{'y' if len(lines)==1 else 'ies'}."
            )
            return
        target_raw = lines[-index]
        print(
            f"[tail-logs] Showing entry #{index} counted from latest (1 = most recent) in {path}"
        )
        render_entry(target_raw, index)
        return

    limit = limit or 1
    if limit <= 0:
        print("[tail-logs] N must be a positive integer.")
        return
    subset = deque(lines, maxlen=limit)
    print(
        f"[tail-logs] Showing last {len(subset)} entr{'y' if len(subset)==1 else 'ies'} from {path}"
    )
    for idx, raw in enumerate(subset, start=1):
        render_entry(raw, idx)


def format_short_history_entry(entry: Dict[str, Any], ordinal: int) -> str:
    """Return a compact textual summary similar to the verbose console footer."""

    separator = "=" * 72
    decision = entry.get("decision") or {}
    preprocess = entry.get("preprocess") or {}
    stages = preprocess.get("stages") or []
    stage_status = " | ".join(
        f"{stage.get('name')}={'OK' if stage.get('success') else 'FAIL'}" for stage in stages
    )
    thought = entry.get("thought_report") or {}
    thought_steps = thought.get("steps") or []
    response = entry.get("response") or {}
    payload = dict(response)
    payload.pop("evidence", None)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    stats = entry.get("stats") or {}
    agent_calls = stats.get("agent_calls") or response.get("agents") or {}
    summary = stats.get("answer_preview") or response.get("answer") or ""
    lines = [
        f"\n--- Entry {ordinal} (short) ---",
        separator,
        "Final Result & Stats",
        separator,
        "Process Flow : Scope --> Preprocess --> Thought Graph --> Final JSON",
        f"Question     : {entry.get('question', '')}",
        f"Scope reply  : {decision.get('reply', '')}",
        f"Preprocess   : {stage_status or 'n/a'}",
        "Thought Graph: "
        f"status={thought.get('status', 'n/a')} nodes={len(thought_steps)} decision={thought.get('decision', '-')}",
        f"Agents       : {agent_calls}",
    ]
    if summary:
        lines.append(f"Answer preview: {summary}")
    lines.append("Payload JSON :")
    lines.append(payload_json)
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser used by ``python main.py``."""
    description = textwrap.dedent(
        """
        Oracle orchestrates the full preprocess + thought-graph pipeline around Envision DSL assets.

        Typical workflows:
          • Batch one question:  python main.py -q "Quels scripts lisent ..."
          • Provide context:     python main.py -q "Combien..." -x "Discussion précédente ..."
          • Live REPL session:   python main.py -i -v
                    • Inspect history:     python main.py -t 5 [-l]
                    • Pinpoint one run:    python main.py -ti 1 --long
        """
    ).strip()
    epilog = "Examples shown above are also available via --help/ -h."
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("-q", "--question", help="Single question to process")
    parser.add_argument("-x", "--context", help="Optional additional context")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive REPL session",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display detailed outputs from the preprocess and thought agents",
    )
    parser.add_argument(
        "-t",
        "--tail-logs",
        type=int,
        metavar="N",
        help="Print the last N history entries (short mode by default)",
    )
    parser.add_argument(
        "-ti",
        "--tail-index",
        type=int,
        metavar="K",
        help="Show only the K-th most recent entry (1 = latest) in short mode",
    )
    parser.add_argument(
        "-l",
        "--long",
        action="store_true",
        help="Display tail output in long JSON form instead of the short summary",
    )
    return parser


def main() -> None:
    """Entry point executed via ``python main.py`` or ``python -m main``."""
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_runtime_config(Path(args.config))
    if args.tail_logs is not None or args.tail_index is not None:
        history_path = resolve_history_file(config)
        tail_history_file(
            history_path,
            limit=args.tail_logs,
            short=not args.long,
            index=args.tail_index,
        )
        return
    agent = MainAgent(config, verbose=args.verbose)

    if args.interactive or not args.question:
        agent.repl()
        return

    answer = agent.handle_question(args.question, context=args.context or "")
    # In verbose mode, JSON is already printed in _print_final_result_stats
    # So we only print it here in non-verbose mode
    if not args.verbose:
        print(answer)


if __name__ == "__main__":
    main()
