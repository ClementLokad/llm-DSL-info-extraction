#!/usr/bin/env python3
"""
Envision Network CLI
====================

Command-line interface for testing the Envision Graph API.
Displays raw API responses with Rich formatting - no post-processing.

What you see is exactly what the API returns.

Usage:
    python -m env_graph.network --build
    python -m env_graph.network --stats
    python -m env_graph.network --tree /
    python -m env_graph.network --read 67982
    python -m env_graph.network --grep "pattern"
    python -m env_graph.network --node 67982
    python -m env_graph.network --search "query"
    python -m env_graph.network --neighbors 67982
    python -m env_graph.network --nodes
    python -m env_graph.network --edges
"""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from env_graph.api import EnvisionGraphAPI

# Initialize
console = Console()
app = typer.Typer(
    name="network",
    help="Envision Network CLI - Test the Graph API with beautiful output.",
    add_completion=False,
    rich_markup_mode="rich",
)


def render_json(data: dict, title: str = "API Response"):
    """Render JSON with Rich syntax highlighting."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    console.print(Panel(
        Syntax(json_str, "json", theme="monokai", word_wrap=True),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


def get_api() -> EnvisionGraphAPI:
    """Get API instance with error handling."""
    try:
        return EnvisionGraphAPI()
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize API:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# Main Callback with Short Flags
# =============================================================================

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # Commands as flags
    build: bool = typer.Option(False, "-b", "--build", help="Build the dependency graph"),
    stats: bool = typer.Option(False, "-s", "--stats", help="Show network statistics"),
    tree: Optional[str] = typer.Option(None, "-t", "--tree", help="Explore folder hierarchy at PATH"),
    read: Optional[str] = typer.Option(None, "-r", "--read", help="Read node content by NODE_ID"),
    grep: Optional[str] = typer.Option(None, "-g", "--grep", help="Search PATTERN in code"),
    node: Optional[str] = typer.Option(None, "-n", "--node", help="Get node metadata by NODE_ID"),
    search: Optional[str] = typer.Option(None, "-q", "--search", help="Search nodes by QUERY"),
    neighbors: Optional[str] = typer.Option(None, "-x", "--neighbors", help="Explore connections of NODE_ID"),
    nodes: bool = typer.Option(False, "-N", "--nodes", help="List all nodes"),
    edges: bool = typer.Option(False, "-E", "--edges", help="List all edges"),
    # Options for tree
    domain: str = typer.Option("scripts", "-d", "--domain", 
                               help="Domain for --tree: scripts, data, or both"),
    depth: int = typer.Option(1, "-D", "--depth", 
                              help="Max depth for --tree (0 = unlimited)"),
    # Options for grep/search/nodes/edges
    types: Optional[str] = typer.Option(None, "--types", 
                                        help="Node types filter (comma-separated)"),
    top_k: int = typer.Option(20, "-k", "--top-k", 
                              help="Max results for --search"),
    # Options for neighbors
    direction: str = typer.Option("all", "--direction", 
                                  help="Direction for --neighbors: incoming, outgoing, all, siblings"),
    relation: Optional[str] = typer.Option(None, "--relation", 
                                           help="Edge type filter for --neighbors"),
    # Options for read
    start_line: Optional[int] = typer.Option(None, "--start", help="Start line for --read"),
    end_line: Optional[int] = typer.Option(None, "--end", help="End line for --read"),
):
    """
    Envision Network CLI - Test the Graph API.
    
    All commands display raw API JSON responses with Rich formatting.
    What you see is exactly what the API returns.
    """
    # Parse types list
    type_list = [t.strip() for t in types.split(",")] if types else None
    
    api = get_api()
    
    try:
        if build:
            console.print("[dim]Building graph...[/dim]")
            result = api.build()
            render_json(result, "build()")
        
        elif stats:
            result = api.get_stats()
            render_json(result, "get_stats()")
        
        elif tree is not None:
            max_depth = depth if depth > 0 else None
            result = api.get_tree(tree, domain=domain, max_depth=max_depth)
            render_json(result, f"get_tree({tree!r}, domain={domain!r}, max_depth={max_depth})")
        
        elif read is not None:
            result = api.read(read, start_line, end_line)
            render_json(result, f"read({read!r}, start_line={start_line}, end_line={end_line})")
        
        elif grep is not None:
            result = api.grep(grep, node_types=type_list)
            render_json(result, f"grep({grep!r}, node_types={type_list})")
        
        elif node is not None:
            result = api.get_node(node)
            render_json(result, f"get_node({node!r})")
        
        elif search is not None:
            result = api.search(search, node_types=type_list, top_k=top_k)
            render_json(result, f"search({search!r}, node_types={type_list}, top_k={top_k})")
        
        elif neighbors is not None:
            result = api.get_neighbors(neighbors, direction=direction, relation_type=relation)
            render_json(result, f"get_neighbors({neighbors!r}, direction={direction!r}, relation_type={relation!r})")
        
        elif nodes:
            node_type = type_list[0] if type_list else None
            result = api.get_nodes(node_type)
            render_json(result, f"get_nodes(node_type={node_type!r})")
        
        elif edges:
            edge_type = type_list[0] if type_list else None
            result = api.get_edges(edge_type)
            render_json(result, f"get_edges(relation_type={edge_type!r})")
        
        else:
            # No command - show help
            console.print(Markdown("""
# Envision Network CLI

Test the Graph API with beautiful JSON output.  
**What you see is exactly what the API returns.**

## Quick Start

```bash
python -m env_graph.network --build
python -m env_graph.network --stats
python -m env_graph.network --tree /
python -m env_graph.network --read 67982
```

## Commands

| Short | Long          | Description                      |
|-------|---------------|----------------------------------|
| `-b`  | `--build`     | Build the dependency graph       |
| `-s`  | `--stats`     | Show network statistics          |
| `-t`  | `--tree`      | Explore folder hierarchy         |
| `-r`  | `--read`      | Read node content                |
| `-g`  | `--grep`      | Search patterns in code          |
| `-n`  | `--node`      | Get node metadata                |
| `-q`  | `--search`    | Search by name/path/id           |
| `-x`  | `--neighbors` | Explore node connections         |
| `-N`  | `--nodes`     | List all nodes                   |
| `-E`  | `--edges`     | List all edges                   |

## Options

| Short | Long          | For             | Description              |
|-------|---------------|-----------------|--------------------------|
| `-d`  | `--domain`    | `--tree`        | scripts, data, or both   |
| `-D`  | `--depth`     | `--tree`        | Max depth (0=unlimited)  |
|       | `--types`     | grep/search/nodes/edges | Node types (comma-sep) |
|       | `--direction` | `--neighbors`   | incoming/outgoing/all/siblings |
|       | `--relation`  | `--neighbors`   | Edge type filter         |
|       | `--start/end` | `--read`        | Line range               |
| `-k`  | `--top-k`     | `--search`      | Max results              |

## Examples

```bash
python -m env_graph.network -t /                           # Tree at root
python -m env_graph.network -t /Clean -d data -D 2         # Data tree, depth 2
python -m env_graph.network -r 67982                       # Read script
python -m env_graph.network -r 67982 --start 1 --end 50    # Lines 1-50
python -m env_graph.network -g "table Items"               # Grep pattern
python -m env_graph.network -x 67982 --direction outgoing  # Outgoing edges only
python -m env_graph.network -N --types script              # List only scripts
python -m env_graph.network -E --types reads               # List only read edges
```

Use `python -m env_graph.network --help` for full options.
"""))
    
    except FileNotFoundError:
        console.print("[yellow]⚠️ No data. Run `python -m env_graph.network --build` first.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e}")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
