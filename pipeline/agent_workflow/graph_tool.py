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
                "Navigate Envision code structure through the dependency graph "
                "(tree, node, neighbors, edges, search)."
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": ["tree", "node", "neighbors", "edges", "search"],
                    "description": "Graph action to execute.",
                },
                "path": {
                    "type": "string",
                    "description": "[tree] Folder path to inspect (default '/').",
                },
                "domain": {
                    "type": "string",
                    "enum": ["scripts", "data", "both"],
                    "description": "[tree] Graph domain.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "[tree] Maximum recursion depth (omit for default).",
                },
                "node_id": {
                    "type": "string",
                    "description": "[node, neighbors] Graph node identifier.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing", "all", "siblings"],
                    "description": "[neighbors] Edge direction perspective.",
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["reads", "writes", "imports", "defines", "contains", "sibling"],
                    "description": "[neighbors, edges] Optional edge type filter.",
                },
                "query": {
                    "type": "string",
                    "description": "[search] Search query on node names/paths.",
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "[search] Optional node type filters.",
                },
            },
            required=["action"],
        )