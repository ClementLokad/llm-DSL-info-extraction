import json
from pathlib import Path
from typing import Any, Dict, Optional

from config_manager import get_config
from env_graph.api import EnvisionGraphAPI
from pipeline.agent_workflow.workflow_base import Tool, _tool_desc


class EnvisionGraphTool(Tool):
    """Tool wrapper around env_graph.EnvisionGraphAPI for structural queries."""

    def __init__(self, config_path: Optional[str] = None, auto_build: bool = False):
        config = get_config()
        self.tool_config = config.get("main_pipeline.graph_tool", {})

        resolved_config_path = (
            config_path
            or self.tool_config.get("config_path")
            or "env_graph/config.yaml"
        )

        self.api = EnvisionGraphAPI(config_path=resolved_config_path)
        self.auto_build = bool(self.tool_config.get("auto_build", auto_build))

    def _execute_once(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "tree": lambda: self.api.get_tree(
                path=args.get("path", "/"),
                domain=args.get(
                    "domain",
                    self.tool_config.get("default_domain", "scripts"),
                ),
                max_depth=args.get("max_depth"),
            ),
            "node": lambda: self.api.get_node(node_id=args["node_id"]),
            "neighbors": lambda: self.api.get_neighbors(
                node_id=args["node_id"],
                direction=args.get("direction", "all"),
                relation_type=args.get("relation_type"),
            ),
            "edges": lambda: self.api.get_edges(
                relation_type=args.get("relation_type")
            ),
            "search": lambda: self.api.search(
                query=args["query"],
                node_types=args.get("node_types"),
            ),
        }

        if action not in handlers:
            raise ValueError(
                f"Unknown graph action: {action}. "
                "Valid actions: tree, node, neighbors, edges, search"
            )

        return handlers[action]()

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute one graph API action and return raw JSON-compatible data."""
        args = dict(kwargs)
        try:
            return self._execute_once(action, args)
        except FileNotFoundError:
            if not self.auto_build:
                raise

            self.api.build()
            return self._execute_once(action, args)

    def to_prompt_text(self, result: Dict[str, Any]) -> str:
        """Serialize result for planner prompt context with a safe size cap."""
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        max_chars = int(self.tool_config.get("response_max_chars", 12000))
        if len(payload) <= max_chars:
            return payload
        return payload[:max_chars] + "\n... [truncated]"

    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="graph_tool",
            description=(
                "Navigate and explore the Envision dependency graph across scripts, data files, "
                "tables, functions, and folders. ALWAYS start with action='tree' on the 'scripts' "
                "domain to understand project structure before deeper navigation. Use execution_order "
                "from folder/file prefixes to reason about pipeline chronology. Then use 'neighbors' "
                "or 'edges' for relationships, 'node' for details, and 'search' ONLY to locate a node "
                "by name/path before switching to a structural action. CRITICAL: search does NOT search "
                "relationship names or code content. It only searches node id, node name, and node path.\n"
                "Typical graph_tool patterns:\n"
                "  • Which scripts import module X? → search X if needed, then neighbors(node_id=X, direction='incoming', relation_type='imports').\n"
                "  • What modules does script X import? → neighbors(node_id=X, direction='outgoing', relation_type='imports').\n"
                "  • What files does script X read/write? → neighbors with relation_type='reads' or 'writes'.\n"
                "  • Global overview of imports/reads/writes → edges(relation_type='imports'|'reads'|'writes').\n"
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": ["tree", "node", "neighbors", "edges", "search"],
                    "description": (
                        "Graph action to execute. 'tree' explores folder hierarchy and should be the "
                        "default first move. 'node' inspects one node. 'neighbors' navigates incoming/"
                        "outgoing/sibling relationships. 'edges' lists relationships by type. 'search' "
                        "finds nodes by name/path only. DO NOT use search for verbs such as 'import', "
                        "'read', 'write', or 'define' because those are edge relations, not node names."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "[tree] Folder path to inspect; default is root '/'. Start at '/' unless you already know the relevant branch.",
                },
                "domain": {
                    "type": "string",
                    "enum": ["scripts", "data", "both"],
                    "description": "[tree] Graph domain to explore. Prefer 'scripts' first; 'data' is useful once you already know the relevant files.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "[tree] Maximum folder recursion depth; omit for auto-limit. Small values are useful for a quick architectural overview.",
                },
                "node_id": {
                    "type": "string",
                    "description": "[node, neighbors] Graph node identifier as returned by tree/search. Can be a script id, function id, table id, or data-file path.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing", "all", "siblings"],
                    "description": (
                        "[neighbors] Think from the perspective of node_id: "
                        "'incoming' = who/what targets this node, "
                        "'outgoing' = what this node targets, "
                        "'all' = both directions, "
                        "'siblings' = same-folder peers with no direction semantics."
                    ),
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["reads", "writes", "imports", "defines", "contains", "sibling"],
                    "description": (
                        "[neighbors, edges] Optional edge type filter. Common patterns: "
                        "scripts that READ a file = node_id is the file + direction='incoming' + relation_type='reads'; "
                        "files that a script READS = node_id is the script + direction='outgoing' + relation_type='reads'; "
                        "scripts that WRITE a file = file + incoming + writes; "
                        "scripts that IMPORT a module = module + incoming + imports; "
                        "modules imported by a script = script + outgoing + imports; "
                        "global import overview = action='edges' with relation_type='imports'."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "[search] Search query on node names or paths ONLY. Good examples: "
                        "'Functions', 'Global Parameters', 'PathSchemas', '/1. utilities/Modules'. "
                        "Bad examples: 'import', 'imports', 'read', 'write' because those are relations "
                        "or code terms, not node names."
                    ),
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "[search] Optional node type filters such as script, data_file, table, function, or folder.",
                },
            },
            required=["action"],
        )
