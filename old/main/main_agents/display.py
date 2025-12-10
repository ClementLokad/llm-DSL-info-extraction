"""
Display utilities for Oracle Agent System
==========================================

Unified display functions for consistent output across all modes.
All output goes through these functions for uniformity.

Modes:
- QUIET: No output (benchmarks)
- SUMMARY: Key milestones only
- VERBOSE: Full traces with truncated outputs
"""

import json
from typing import Any, Dict, List, Optional

from .models import (
    OutputMode,
    ScopeDecision,
    ScopeStatus,
    StepResult,
    ExecStats,
    WorkingMemory,
    OracleResult,
    ThoughtNode,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Max lines to show for agent outputs (first N/2 + last N/2)
DEFAULT_MAX_LINES = 20

# Box drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_T = "┬"
BOX_B = "┴"
BOX_L = "├"
BOX_R = "┤"
BOX_X = "┼"

# Status symbols
SYM_OK = "✓"
SYM_FAIL = "✗"
SYM_WARN = "⚠"
SYM_INFO = "ℹ"
SYM_ARROW = "→"
SYM_BULLET = "•"


# =============================================================================
# CORE OUTPUT FUNCTIONS
# =============================================================================

class Display:
    """Unified display handler respecting output mode."""
    
    def __init__(self, mode: OutputMode = OutputMode.SUMMARY):
        self.mode = mode
        self.max_lines = DEFAULT_MAX_LINES
    
    def _print(self, *args, **kwargs) -> None:
        """Print only if not QUIET mode."""
        if self.mode != OutputMode.QUIET:
            print(*args, **kwargs)
    
    def _verbose(self, *args, **kwargs) -> None:
        """Print only in VERBOSE mode."""
        if self.mode == OutputMode.VERBOSE:
            print(*args, **kwargs)
    
    # =========================================================================
    # SECTIONS & HEADERS
    # =========================================================================
    
    def header(self, title: str, char: str = "=", width: int = 70) -> None:
        """Print a section header."""
        self._print()
        self._print(char * width)
        self._print(title.center(width))
        self._print(char * width)
    
    def subheader(self, title: str, char: str = "-", width: int = 70) -> None:
        """Print a subsection header."""
        self._print()
        self._print(char * width)
        self._print(title)
        self._print(char * width)
    
    def separator(self, char: str = "-", width: int = 70) -> None:
        """Print a separator line."""
        self._print(char * width)
    
    # =========================================================================
    # KEY-VALUE DISPLAY
    # =========================================================================
    
    def kv(self, key: str, value: Any, indent: int = 0) -> None:
        """Print key-value pair with alignment."""
        prefix = " " * indent
        key_str = f"{key:<15}"
        self._print(f"{prefix}{key_str}: {value}")
    
    def kv_verbose(self, key: str, value: Any, indent: int = 0) -> None:
        """Print key-value only in VERBOSE mode."""
        if self.mode == OutputMode.VERBOSE:
            self.kv(key, value, indent)
    
    # =========================================================================
    # STATUS INDICATORS
    # =========================================================================
    
    def status(self, agent: str, success: bool, summary: str) -> None:
        """Print agent status line."""
        sym = SYM_OK if success else SYM_FAIL
        status_word = "success" if success else "failed"
        self._print(f"  [{agent}] {sym} {status_word}: {summary[:60]}")
    
    def milestone(self, text: str) -> None:
        """Print a milestone marker."""
        self._print(f"\n{SYM_ARROW} {text}")
    
    def bullet(self, text: str, indent: int = 2) -> None:
        """Print a bullet point."""
        prefix = " " * indent
        self._print(f"{prefix}{SYM_BULLET} {text}")
    
    # =========================================================================
    # AGENT OUTPUT DISPLAY
    # =========================================================================
    
    def agent_start(self, agent: str, task: str) -> None:
        """Announce agent starting."""
        self._verbose(f"\n[{agent.upper()}] {task[:80]}{'...' if len(task) > 80 else ''}")
    
    def agent_result(self, result: StepResult) -> None:
        """Display agent result with truncated output."""
        if self.mode == OutputMode.QUIET:
            return
        
        # Status line (always in SUMMARY+)
        sym = SYM_OK if result.success else SYM_FAIL
        status = "success" if result.success else "failed"
        
        # Node info
        node_info = f"[Node {result.node_index}]" if result.node_index else ""
        depth_info = f"(depth={result.depth})" if result.depth > 0 else ""
        attempts_info = f"*{result.attempts}" if result.attempts > 1 else ""
        
        self._print(f"\n{node_info} {result.agent}{attempts_info} {depth_info}")
        self._print(f"  {sym} {status}: {result.summary[:70]}")
        
        # Verbose: show truncated output
        if self.mode == OutputMode.VERBOSE and result.output:
            self._print_output(result.output)
    
    def _print_output(self, output: Dict[str, Any], max_lines: int = None) -> None:
        """Print output dict with line truncation."""
        if max_lines is None:
            max_lines = self.max_lines
        
        try:
            text = json.dumps(output, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(output)
        
        lines = text.split("\n")
        
        if len(lines) <= max_lines:
            for line in lines:
                self._print(f"    {line}")
        else:
            # Show first N/2 and last N/2
            half = max_lines // 2
            for line in lines[:half]:
                self._print(f"    {line}")
            self._print(f"    ... [{len(lines) - max_lines} lines omitted] ...")
            for line in lines[-half:]:
                self._print(f"    {line}")
    
    # =========================================================================
    # SCOPE DECISION
    # =========================================================================
    
    def scope_decision(self, decision: ScopeDecision) -> None:
        """Display scope classification result."""
        self.subheader("Scope Classification")
        
        status_map = {
            ScopeStatus.GREETING: ("Greeting", SYM_INFO),
            ScopeStatus.PROJECT: ("In Scope", SYM_OK),
            ScopeStatus.OFF_TOPIC: ("Off Topic", SYM_WARN),
        }
        
        label, sym = status_map.get(decision.status, ("Unknown", "?"))
        self._print(f"  Status: {sym} {label}")
        self._print(f"  Reason: {decision.reason}")
        
        if decision.status != ScopeStatus.PROJECT:
            self._print(f"  Reply: {decision.reply}")
    
    # =========================================================================
    # THOUGHT GRAPH
    # =========================================================================
    
    def thinking_iteration(
        self,
        iteration: int,
        max_iter: int,
        node: str,
        depth: int,
        max_depth: int,
        thinking: str,
        decision: str,
        action_type: str = ""
    ) -> None:
        """Display Thinker iteration."""
        self._verbose()
        self._verbose(f"[Thinker] Iteration {iteration}/{max_iter} | Node: {node} | Depth: {depth}/{max_depth}")
        
        if self.mode == OutputMode.VERBOSE:
            self._print(f"  Thinking: {thinking[:150]}{'...' if len(thinking) > 150 else ''}")
            
            decision_str = decision
            if action_type:
                decision_str = f"{decision} {SYM_ARROW} {action_type}"
            self._print(f"  Decision: {decision_str}")
    
    def thought_tree(self, memory: WorkingMemory) -> None:
        """Display the thought tree."""
        self.subheader("Thought Tree")
        
        tree_str = self._render_tree(memory)
        for line in tree_str.split("\n"):
            self._print(f"  {line}")
    
    def _render_tree(self, memory: WorkingMemory) -> str:
        """Render thought tree as ASCII."""
        if not memory.nodes:
            return "(empty)"
        
        lines = ["root"]
        
        def render_node(index: str, prefix: str, is_last: bool) -> None:
            node = memory.nodes.get(index)
            if not node:
                return
            
            connector = "└─" if is_last else "├─"
            attempts_str = f" *{node.attempts}" if node.attempts > 1 else ""
            lines.append(f"{prefix}{connector} {index} ({node.agent}{attempts_str})")
            
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = node.children
            for i, child in enumerate(children):
                render_node(child, child_prefix, i == len(children) - 1)
        
        for i, root_child in enumerate(memory.root_children):
            render_node(root_child, "", i == len(memory.root_children) - 1)
        
        return "\n".join(lines)
    
    # =========================================================================
    # EXECUTION STATS
    # =========================================================================
    
    def exec_stats(self, stats: ExecStats, config_limits: Dict[str, int] = None) -> None:
        """Display execution statistics."""
        self.subheader("Execution Statistics")
        
        limits = config_limits or {}
        
        self.kv("Iterations", f"{stats.iterations} / {limits.get('max_iterations', '?')}")
        self.kv("Total Steps", f"{stats.total_steps} / {limits.get('max_total_steps', '?')}")
        self.kv("Max Depth", f"{stats.max_depth} / {limits.get('max_depth', '?')}")
        self.kv("Retries", stats.retries)
        self.kv("Splits", stats.splits)
        
        if stats.agent_calls:
            self._print()
            self._print("  Agent calls:")
            for agent, count in sorted(stats.agent_calls.items()):
                self._print(f"    {agent}: {count}")
    
    # =========================================================================
    # HYPOTHESES & PATTERNS
    # =========================================================================
    
    def hypotheses(self, memory: WorkingMemory) -> None:
        """Display current hypotheses."""
        if not memory.hypotheses:
            return
        
        self._verbose()
        self._verbose(f"  Hypotheses ({len(memory.hypotheses)}):")
        for h in memory.hypotheses:
            self._verbose(f"    [{h.confidence:.2f}] {h.claim[:80]}")
    
    def patterns(self, memory: WorkingMemory) -> None:
        """Display discovered patterns."""
        if not memory.patterns:
            return
        
        self._verbose()
        self._verbose(f"  Patterns ({len(memory.patterns)}):")
        for p in list(memory.patterns)[:5]:
            self._verbose(f"    {SYM_BULLET} {p}")
    
    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    
    def final_result(self, result: OracleResult) -> None:
        """Display final Oracle result."""
        self.header("Oracle Response")
        
        self.kv("Question", result.question)
        self.kv("Confidence", f"{result.confidence:.2f}")
        self.kv("Time", f"{result.time_ms:.0f}ms")
        
        if result.stats:
            agents_str = ", ".join(f"{k}*{v}" for k, v in result.stats.agent_calls.items())
            self.kv("Agents", agents_str or "none")
        
        self._print()
        self.separator()
        self._print("Answer:")
        self.separator()
        self._print()
        self._print(result.answer)
        self._print()
        self.separator()
        
        # References
        if result.synth and result.synth.references:
            self._print()
            self._print(f"References ({len(result.synth.references)}):")
            for ref in result.synth.references[:10]:
                self.bullet(ref)
    
    # =========================================================================
    # ERROR DISPLAY
    # =========================================================================
    
    def error(self, message: str, details: str = None) -> None:
        """Display error message."""
        self._print(f"\n{SYM_FAIL} Error: {message}")
        if details:
            self._print(f"  Details: {details}")
    
    def warning(self, message: str) -> None:
        """Display warning message."""
        self._print(f"{SYM_WARN} Warning: {message}")
