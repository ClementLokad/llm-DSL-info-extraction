#!/usr/bin/env python3
"""
Viewer for benchmark results and statistics JSON files.

Reconstructs the Rich terminal display from JSON exports of:
- Benchmark results (with models, results, token counts)
- Stats (tool calls, LLM timing, etc.)

Usage:
    python view_benchmark_results.py results.json
    python view_benchmark_results.py stats.json --quiet
    python view_benchmark_results.py results.json --models --tokens
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.align import Align
from rich.markup import escape


def detect_json_format(data: Dict[str, Any]) -> str:
    """Detect whether JSON is benchmark results or stats format.
    
    Returns: "benchmark", "stats", or "unknown"
    """
    if "Results" in data and isinstance(data.get("Results"), list):
        return "benchmark"
    elif "total_benchmark_time" in data and "llm_generation_times" in data:
        return "stats"
    elif "tool_call_counts" in data and "llm_generation_times" in data:
        return "stats"
    else:
        return "unknown"


def display_benchmark_results(
    console: Console,
    data: Dict[str, Any],
    quiet: bool = False,
    show_models: bool = False,
    show_tokens: bool = False,
) -> None:
    """Display benchmark results in Rich format matching main.py style."""
    
    grades = data.get("Results", [])
    if not grades:
        console.print("[yellow]No results to display.[/yellow]")
        return
    
    # Title
    console.print(Markdown("# Benchmark Results"))
    
    # Detailed results first (unless --quiet)
    if not quiet:
        for r in grades:
            console.print(f"\n[bold green]{r.get('id', '?')}) Question: {escape(r.get('question', ''))} [/bold green]\n")
            console.print(f"[bold purple]  Référence: {escape(r.get('reference', ''))}[/bold purple]")
            console.print("\n[bold blue]  LLM: [/bold blue]")
            console.print(Markdown(escape(r.get('llm_response', ''))))
            console.print(f"\n[bold red] Score : [/bold red]{escape(str(r.get('score', 'N/A')))}")
            if "reasoning" in r:
                console.print(f"\n[bold gold3]  LLM Judge reasoning:[/bold gold3]\n{escape(r['reasoning'])}")
    
    # Summary table (after details)
    table = Table(title="Benchmark Grades", show_lines=True)
    table.add_column("Question", style="cyan", no_wrap=False)
    table.add_column("Score", style="magenta")
    
    for r in grades:
        table.add_row(f"{r.get('id', '?')}) " + r.get('question', ''), f"{r.get('score', 0):.4f}")
    
    console.print("\n")
    console.print(Align.center(table))
    
    # Average score
    if grades:
        avg = sum(r.get("score", 0) for r in grades) / len(grades)
        console.print(f"\n[bold]Moyenne globale : {avg:.4f}[/bold]")
        console.print(f"[dim]Questions justes: {len([r for r in grades if r.get('score', 0) > 0])}[/dim]")
    
    # Models info
    if show_models and "Models" in data:
        console.print("\n[bold]Models used:[/bold]")
        models = data["Models"]
        for role, model in models.items():
            console.print(f"  • {role}: [cyan]{model}[/cyan]")
    
    # Token usage
    if show_tokens and "Tokens used" in data:
        tokens = data["Tokens used"]
        console.print(f"\n[bold]Tokens used:[/bold] {tokens.get('In', 0)} [green]in[/green], {tokens.get('Out', 0)} [red]out[/red]")
    
    # File metadata
    if "Timestamp" in data:
        console.print(f"[dim]Timestamp: {data['Timestamp']}[/dim]")


def display_stats(
    console: Console,
    data: Dict[str, Any],
    quiet: bool = False,
) -> None:
    """Display stats in markdown table format (matching stats_reporter.py style)."""
    
    console.print(Markdown("# Benchmark Statistics"))
    
    total_time = data.get("total_benchmark_time")
    if total_time is not None:
        console.print(Markdown(f"\n## ⏱️ Total Benchmark Time: **{total_time:.2f}s**"))
    
    # Tool call counts
    tool_counts = data.get("tool_call_counts", {})
    if tool_counts:
        console.print("\n## 🔧 Tool Call Counts")
        console.print("\n| Tool | Count |")
        console.print("|:------:|:-------:|")
        for tool_name in sorted(tool_counts.keys()):
            count = tool_counts[tool_name]
            console.print(f"| `{tool_name}` | **{count}** |")
    else:
        console.print("\n## 🔧 Tool Call Counts\n\nNo tool calls recorded.")
    
    # LLM generation times
    llm_times = data.get("llm_generation_times", {})
    if llm_times:
        console.print("\n## 🤖 LLM Generation Times")
        console.print("\n| Role | Duration | % of Total |")
        console.print("|:------:|:----------:|:-----------:|")
        for role in ["planner", "solver", "cleaning", "distillation", "grader"]:
            elapsed = llm_times.get(role, 0)
            pct = ""
            if total_time and total_time > 0:
                pct = f"{100 * elapsed / total_time:.1f}"
            else:
                pct = "0.0"
            console.print(f"| {role} | **{elapsed:.2f}s** | {pct}% |")
    
    # Tool execution times
    tool_exec = data.get("tool_execution_times", {})
    if tool_exec:
        console.print("\n## ⚙️ Tool Execution Times")
        console.print("\n| Tool | Duration | % of Total |")
        console.print("|:------:|:----------:|:-----------:|")
        for tool_name in sorted(tool_exec.keys()):
            elapsed = tool_exec[tool_name]
            pct = ""
            if total_time and total_time > 0:
                pct = f"{100 * elapsed / total_time:.1f}"
            else:
                pct = "0.0"
            console.print(f"| `{tool_name}` | **{elapsed:.2f}s** | {pct}% |")
    else:
        console.print("\n## ⚙️ Tool Execution Times\n\nNo tool executions recorded.")
    
    # Rate limit delay
    rate_limit = data.get("rate_limit_delay_total", 0)
    if total_time and total_time > 0:
        pct = f"{100 * rate_limit / total_time:.1f}"
    else:
        pct = "0.0"
    console.print(f"\n## ⏳ Rate Limit Delay")
    console.print(f"\n**{rate_limit:.2f}s** ({pct}% of total time)")
    
    # Token usage
    token_stats = data.get("token_stats")
    if token_stats:
        tokens_in = token_stats.get("tokens_in", 0)
        tokens_out = token_stats.get("tokens_out", 0)
        console.print("\n## 🔤 Token Usage")
        console.print(f"\n- **Tokens In:** `{tokens_in:,}`")
        console.print(f"- **Tokens Out:** `{tokens_out:,}`")


def load_and_display(
    filepath: str,
    console: Console,
    quiet: bool = False,
    show_models: bool = False,
    show_tokens: bool = False,
) -> int:
    """Load JSON and display appropriately. Returns 0 on success, 1 on error."""
    
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
    
    # Detect format
    fmt = detect_json_format(data)
    
    if fmt == "benchmark":
        display_benchmark_results(console, data, quiet=quiet, show_models=show_models, show_tokens=show_tokens)
    elif fmt == "stats":
        display_stats(console, data, quiet=quiet)
    else:
        console.print("[bold red]Error:[/bold red] Unknown JSON format.")
        console.print("Expected either:")
        console.print("  - Benchmark results: { 'Models': {...}, 'Results': [...] }")
        console.print("  - Stats: { 'total_benchmark_time': ..., 'llm_generation_times': {...} }")
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View benchmark results or statistics from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # View benchmark results with full details
  python view_benchmark_results.py data/benchmark_results/results.json
  
  # View only summary table
  python view_benchmark_results.py results.json --quiet
  
  # View benchmark with models and token info highlighted
  python view_benchmark_results.py results.json --models --tokens
  
  # View statistics
  python view_benchmark_results.py data/Statistics/stats_2026-04-17.json
        """
    )
    
    parser.add_argument(
        "filepath",
        help="Path to JSON file (benchmark results or stats)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Display only summary table, skip detailed results"
    )
    
    parser.add_argument(
        "--models",
        action="store_true",
        help="Show models section (for benchmark results)"
    )
    
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Highlight token usage section"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    exit_code = load_and_display(
        args.filepath,
        console,
        quiet=args.quiet,
        show_models=args.models,
        show_tokens=args.tokens,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
