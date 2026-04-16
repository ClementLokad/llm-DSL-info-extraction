"""
StatsReporter - Formats and saves benchmark statistics.

Provides functions to generate human-readable reports and save stats to JSON files.
"""

import json
import os
from datetime import datetime
from typing import Optional

from pipeline.stats_collector import BenchmarkStatsCollector


def format_stats_report(
    stats: BenchmarkStatsCollector,
    include_tokens: bool = False,
    token_stats: Optional[dict] = None,
) -> str:
    """Format collected statistics into a human-readable Markdown report.

    Args:
        stats: The BenchmarkStatsCollector instance.
        include_tokens: Whether to include token counts in the report.
        token_stats: Optional dict with keys 'tokens_in' and 'tokens_out'.

    Returns:
        Formatted report as Markdown string.
    """
    lines = ["# Benchmark Statistics"]

    report = stats.get_report()
    total_time = report["total_benchmark_time"]

    # Overall timing - highlighted
    if total_time is not None:
        lines.append(f"\n## ⏱️ Total Benchmark Time: **{total_time:.2f}s**")

    # Tool call counts
    if report["tool_call_counts"]:
        lines.append("\n## 🔧 Tool Call Counts")
        lines.append("\n| Tool | Count |")
        lines.append("|------|-------|")
        for tool_name in sorted(report["tool_call_counts"].keys()):
            count = report["tool_call_counts"][tool_name]
            lines.append(f"| `{tool_name}` | **{count}** |")
    else:
        lines.append("\n## 🔧 Tool Call Counts\n\nNo tool calls recorded.")

    # LLM generation times - table format
    lines.append("\n## 🤖 LLM Generation Times")
    lines.append("\n| Role | Duration | % of Total |")
    lines.append("|------|----------|-----------|")
    for role in ["planner-followup-call", "planner-initial-call", "planner-answer-analysis",
                 "solver", "cleaning", "distillation", "grader"]:
        elapsed = report["llm_generation_times"].get(role, 0)
        pct = ""
        if total_time and total_time > 0:
            pct = f"{100 * elapsed / total_time:.1f}"
        else:
            pct = "0.0"
        lines.append(f"| {role} | **{elapsed:.2f}s** | {pct}% |")

    # Tool execution times - table format
    if report["tool_execution_times"]:
        lines.append("\n## ⚙️ Tool Execution Times")
        lines.append("\n| Tool | Duration | % of Total |")
        lines.append("|------|----------|-----------|")
        for tool_name in sorted(report["tool_execution_times"].keys()):
            elapsed = report["tool_execution_times"][tool_name]
            pct = ""
            if total_time and total_time > 0:
                pct = f"{100 * elapsed / total_time:.1f}"
            else:
                pct = "0.0"
            lines.append(f"| `{tool_name}` | **{elapsed:.2f}s** | {pct}% |")
    else:
        lines.append("\n## ⚙️ Tool Execution Times\n\nNo tool executions recorded.")

    # Rate limit delay
    rate_limit_delay = report["rate_limit_delay_total"]
    if total_time and total_time > 0:
        pct = f"{100 * rate_limit_delay / total_time:.1f}"
    else:
        pct = "0.0"
    lines.append(f"\n## ⏳ Rate Limit Delay")
    lines.append(f"\n**{rate_limit_delay:.2f}s** ({pct}% of total time)")

    # Token counts if requested
    if include_tokens and token_stats:
        tokens_in = token_stats.get("tokens_in", 0)
        tokens_out = token_stats.get("tokens_out", 0)
        lines.append("\n## 🔤 Token Usage")
        lines.append(f"\n- **Tokens In:** `{tokens_in:,}`")
        lines.append(f"- **Tokens Out:** `{tokens_out:,}`")

    return "\n".join(lines)


def save_stats_to_json(
    stats: BenchmarkStatsCollector,
    output_dir: str = "data/benchmark_results",
    include_tokens: bool = False,
    token_stats: Optional[dict] = None,
) -> str:
    """Save collected statistics to a timestamped JSON file.

    Args:
        stats: The BenchmarkStatsCollector instance.
        output_dir: Directory where the JSON file will be saved.
        include_tokens: Whether to include token counts in the output.
        token_stats: Optional dict with keys 'tokens_in' and 'tokens_out'.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"stats_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    report = stats.get_report()

    # Build output JSON
    output = {
        "timestamp": timestamp,
        "stats": report,
    }

    if include_tokens and token_stats:
        output["tokens"] = token_stats

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath
