"""
BenchmarkStatsCollector - Collects timing and tool call statistics for benchmark runs.

Tracks:
- Tool call counts (per tool type)
- LLM generation timing (per role: planner, solver, cleaning, distillation)
- Tool execution timing (per tool)
- Rate limit delay total
- Overall benchmark timing
"""

import time
from typing import Dict, Optional

from utils.config_manager import ConfigManager


class BenchmarkStatsCollector:
    """Collects and aggregates timing and tool call statistics during benchmark execution."""

    def __init__(self):
        """Initialize the stats collector."""
        self.tool_call_counts: Dict[str, int] = {}
        self.llm_generation_times: Dict[str, float] = {
            "planner-followup-call": 0.0, # subsequent planner calls after the initial tool selection
            "planner-initial-call": 0.0, # the initial planner tool-selection call but at each turn
            "planner-answer-analysis": 0.0,
            "solver": 0.0,
            "cleaning": 0.0,
            "distillation": 0.0,
            "grader": 0.0,
        }
        self.tool_execution_times: Dict[str, float] = {}
        self.rate_limit_delay_total: float = 0.0

        self.benchmark_start_time: Optional[float] = None
        self.benchmark_end_time: Optional[float] = None

        # For tracking active LLM generation timing
        self._llm_start_times: Dict[str, float] = {}
        # For tracking active tool execution timing
        self._tool_start_times: Dict[str, float] = {}

    def start_benchmark(self) -> None:
        """Mark the start of the benchmark."""
        self.benchmark_start_time = time.perf_counter()

    def end_benchmark(self) -> None:
        """Mark the end of the benchmark."""
        self.benchmark_end_time = time.perf_counter()

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call for the given tool name.

        Args:
            tool_name: Name of the tool being called (e.g., 'rag_tool', 'grep_tool').
        """
        if tool_name not in self.tool_call_counts:
            self.tool_call_counts[tool_name] = 0
        self.tool_call_counts[tool_name] += 1

    def start_llm_generation(self, role: str) -> None:
        """Start timing an LLM generation for the given role.

        Args:
            role: The role of the LLM (e.g., 'planner-followup-call', 'planner-initial-call', 'planner-answer-analysis', 'solver', 'cleaning', 'distillation', 'grader').
        """
        self._llm_start_times[role] = time.perf_counter()

    def end_llm_generation(self, role: str) -> None:
        """End timing an LLM generation for the given role and accumulate the time.

        Args:
            role: The role of the LLM (e.g., 'planner-followup-call', 'planner-initial-call', 'planner-answer-analysis', 'solver', 'cleaning', 'distillation', 'grader').
        """
        if role not in self._llm_start_times:
            return  # start_llm_generation was not called

        elapsed = time.perf_counter() - self._llm_start_times[role]
        self.llm_generation_times[role] += elapsed
        del self._llm_start_times[role]

    def start_tool_execution(self, tool_name: str) -> None:
        """Start timing a tool execution for the given tool name.

        Args:
            tool_name: Name of the tool (e.g., 'rag_tool', 'grep_tool').
        """
        self._tool_start_times[tool_name] = time.perf_counter()

    def end_tool_execution(self, tool_name: str) -> None:
        """End timing a tool execution for the given tool name and accumulate the time.

        Args:
            tool_name: Name of the tool (e.g., 'rag_tool', 'grep_tool').
        """
        if tool_name not in self._tool_start_times:
            return  # start_tool_execution was not called

        elapsed = time.perf_counter() - self._tool_start_times[tool_name]
        if tool_name not in self.tool_execution_times:
            self.tool_execution_times[tool_name] = 0.0
        self.tool_execution_times[tool_name] += elapsed
        del self._tool_start_times[tool_name]

    def record_rate_limit_delay(self, delay_amount: float) -> None:
        """Record time spent in rate limit delays.

        Args:
            delay_amount: The delay duration in seconds.
        """
        self.rate_limit_delay_total += delay_amount

    def get_report(self) -> Dict:
        """Get a dictionary representation of all collected statistics.

        Returns:
            Dictionary containing all stats.
        """
        total_benchmark_time = None
        if self.benchmark_start_time and self.benchmark_end_time:
            total_benchmark_time = self.benchmark_end_time - self.benchmark_start_time

        return {
            "total_benchmark_time": total_benchmark_time,
            "tool_call_counts": self.tool_call_counts,
            "llm_generation_times": self.llm_generation_times,
            "tool_execution_times": self.tool_execution_times,
            "rate_limit_delay_total": self.rate_limit_delay_total,
            "benchmark_start_time": self.benchmark_start_time,
            "benchmark_end_time": self.benchmark_end_time,
        }

# Global configuration instance
_collector_instance: Optional[BenchmarkStatsCollector] = None

def get_collector() -> BenchmarkStatsCollector:
    """
    Get global benchmark stats collector instance (singleton pattern).
    
    Returns:
        BenchmarkStatsCollector instance
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = BenchmarkStatsCollector()
    return _collector_instance

def reload_collector() -> BenchmarkStatsCollector:
    """
    Reload benchmark stats collector (useful for testing or configuration changes).

    Returns:
        New BenchmarkStatsCollector instance
    """
    global _collector_instance
    _collector_instance = BenchmarkStatsCollector()
    return _collector_instance