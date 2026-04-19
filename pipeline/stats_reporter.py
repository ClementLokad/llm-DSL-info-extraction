"""
StatsReporter - Formats and saves benchmark statistics.

Provides functions to generate human-readable reports and save stats to JSON files.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align

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


def display_stats_from_dict(
    console: Console,
    stats_dict: Dict[str, Any],
    include_tokens: bool = True,
) -> None:
    """Display statistics from a dictionary (loaded from JSON).
    
    Args:
        console: Rich Console instance for output.
        stats_dict: Dictionary containing stats data (e.g., from JSON).
        include_tokens: Whether to display token information.
    """
    # Extract stats and token data
    report = stats_dict if isinstance(stats_dict, dict) else {}
    total_time = report.get("total_benchmark_time")
    
    # Build markdown output similar to format_stats_report
    lines = ["# Benchmark Statistics"]
    
    # Overall timing
    if total_time is not None:
        lines.append(f"\n## ⏱️ Total Benchmark Time: **{total_time:.2f}s**")
    
    # Tool call counts
    tool_counts = report.get("tool_call_counts", {})
    if tool_counts:
        lines.append("\n## 🔧 Tool Call Counts")
        lines.append("\n| Tool | Count |")
        lines.append("|:------:|:-------:|")
        for tool_name in sorted(tool_counts.keys()):
            count = tool_counts[tool_name]
            lines.append(f"| `{tool_name}` | **{count}** |")
    else:
        lines.append("\n## 🔧 Tool Call Counts\n\nNo tool calls recorded.")
    
    # LLM generation times
    llm_times = report.get("llm_generation_times", {})
    if llm_times:
        lines.append("\n## 🤖 LLM Generation Times")
        lines.append("\n| Role | Duration | % of Total |")
        lines.append("|:------:|:----------:|:-----------:|")
        for role in ["planner-followup-call", "planner-initial-call", "planner-answer-analysis",
                     "solver", "cleaning", "distillation", "grader"]:
            elapsed = llm_times.get(role, 0)
            pct = ""
            if total_time and total_time > 0:
                pct = f"{100 * elapsed / total_time:.1f}"
            else:
                pct = "0.0"
            lines.append(f"| {role} | **{elapsed:.2f}s** | {pct}% |")
    
    # Tool execution times
    tool_exec = report.get("tool_execution_times", {})
    if tool_exec:
        lines.append("\n## ⚙️ Tool Execution Times")
        lines.append("\n| Tool | Duration | % of Total |")
        lines.append("|:------:|:----------:|:-----------:|")
        for tool_name in sorted(tool_exec.keys()):
            elapsed = tool_exec[tool_name]
            pct = ""
            if total_time and total_time > 0:
                pct = f"{100 * elapsed / total_time:.1f}"
            else:
                pct = "0.0"
            lines.append(f"| `{tool_name}` | **{elapsed:.2f}s** | {pct}% |")
    else:
        lines.append("\n## ⚙️ Tool Execution Times\n\nNo tool executions recorded.")
    
    # Rate limit delay
    rate_limit = report.get("rate_limit_delay_total", 0)
    if total_time and total_time > 0:
        pct = f"{100 * rate_limit / total_time:.1f}"
    else:
        pct = "0.0"
    lines.append(f"\n## ⏳ Rate Limit Delay")
    lines.append(f"\n**{rate_limit:.2f}s** ({pct}% of total time)")
    
    # Display
    markdown_output = "\n".join(lines)
    console.print(Align.center(Markdown(markdown_output)))


def load_and_display_stats(filepath: str, console: Console) -> int:
    """Load stats from JSON file and display in terminal.
    
    Args:
        filepath: Path to the JSON stats file.
        console: Rich Console instance.
        
    Returns:
        0 on success, 1 on error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] File not found: {filepath}")
        return 1
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON format: {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1
    
    # Handle both old and new JSON formats
    if isinstance(data, dict) and "stats" in data:
        # New format: { "timestamp": ..., "stats": {...}, "tokens": {...} }
        stats_dict = data.get("stats", {})
    elif isinstance(data, dict) and "total_benchmark_time" in data:
        # Direct stats dict
        stats_dict = data
    else:
        console.print("[bold red]Error:[/bold red] Unknown JSON format for stats.")
        console.print("Expected format with 'total_benchmark_time' and 'llm_generation_times'.")
        return 1
    
    # Display stats
    display_stats_from_dict(console, stats_dict, include_tokens=True)
    
    # Display file info
    if "timestamp" in data:
        console.print(f"\n[dim]Timestamp: {data['timestamp']}[/dim]")
    
    return 0


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="View benchmark statistics from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # View stats from file
  python -m pipeline.stats_reporter data/Statistics/stats_2026-04-17_10-30-45.json
  
  # Or using direct invocation
  python pipeline/stats_reporter.py data/Statistics/stats_2026-04-17_10-30-45.json
        """
    )
    
    parser.add_argument(
        "filepath",
        help="Path to JSON stats file"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    exit_code = load_and_display_stats(args.filepath, console)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
