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

    def validate_graph_arguments(self, action: str, arguments: Dict[str, Any]) -> tuple:
        """Validate and normalize graph_tool arguments.
        
        Returns: (is_valid: bool, error_msg: str, normalized_args: dict)
        """
        if not action:
            return False, "action parameter is required", {}
        
        # Normalize common variants
        action_normalized = action.strip().lower()
        normalization_map = {
            "nodes": "node",
            "edge": "edges",
            "neighbour": "neighbors",
            "neighbours": "neighbors",
        }
        if action_normalized in normalization_map:
            action_normalized = normalization_map[action_normalized]
        
        # Validate action is in whitelist
        valid_actions = ["tree", "node", "search", "neighbors", "edges"]
        if action_normalized not in valid_actions:
            return (
                False,
                f"Unknown graph action: '{action}'. Valid actions: {', '.join(valid_actions)}",
                {}
            )
        
        # Validate required and optional arguments per action
        normalized_args = dict(arguments)
        normalized_args["action"] = action_normalized
        
        # Action-specific validation
        if action_normalized == "node":
            if "node_id" not in arguments or not arguments["node_id"]:
                return False, "action='node' requires node_id parameter", {}
        
        elif action_normalized == "neighbors":
            if "node_id" not in arguments or not arguments["node_id"]:
                return False, "action='neighbors' requires node_id parameter", {}
            # Validate direction enum
            if "direction" in arguments:
                valid_directions = ["incoming", "outgoing", "all", "siblings"]
                if arguments["direction"] not in valid_directions:
                    return (
                        False,
                        f"Invalid direction: '{arguments['direction']}'. Valid: {', '.join(valid_directions)}",
                        {}
                    )
            # Validate relation_type enum if provided
            if "relation_type" in arguments:
                valid_relations = ["reads", "writes", "imports", "defines", "contains", "sibling"]
                if arguments["relation_type"] not in valid_relations:
                    return (
                        False,
                        f"Invalid relation_type: '{arguments['relation_type']}'. Valid: {', '.join(valid_relations)}",
                        {}
                    )
        
        elif action_normalized == "search":
            if "query" not in arguments or not arguments["query"]:
                return False, "action='search' requires query parameter", {}
            # Validate node_types enum if provided
            if "node_types" in arguments:
                valid_types = ["script", "data_file", "table", "function", "folder"]
                provided_types = arguments["node_types"]
                if isinstance(provided_types, list):
                    for nt in provided_types:
                        if nt not in valid_types:
                            return (
                                False,
                                f"Invalid node_type: '{nt}'. Valid: {', '.join(valid_types)}",
                                {}
                            )
        
        elif action_normalized == "tree":
            # tree is permissive, optional path and domain
            if "domain" in arguments:
                valid_domains = ["scripts", "data", "both"]
                if arguments["domain"] not in valid_domains:
                    return (
                        False,
                        f"Invalid domain: '{arguments['domain']}'. Valid: {', '.join(valid_domains)}",
                        {}
                    )
            # Validate max_depth is int if provided
            if "max_depth" in arguments:
                try:
                    int(arguments["max_depth"])
                except (ValueError, TypeError):
                    return False, f"max_depth must be an integer, got: {arguments['max_depth']}", {}
        
        # All validations passed
        return True, "", normalized_args

    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="graph_tool",
            description=(
                "Navigate the Envision dependency graph (scripts, data files, tables, functions, folders). "
                "Nodes are identified by their logical paths (e.g., '/1. utilities/Modules/Functions.nvn').\n\n"
                "MANDATORY WORKFLOW:\n"
                "1. Start with action='tree' to grasp the folder/file hierarchy.\n"
                "2. (Optional) Use 'search' or 'node' to locate specific targets by name or ID.\n"
                "3. Conclude with 'neighbors' or 'edges' to extract relationships.\n\n"
                "CONTROL FLOW:\n"
                "- Actions 'tree', 'node', and 'search' return control to the PLANNER for further reasoning.\n"
                "- Actions 'neighbors' and 'edges' immediately send their results to the SOLVER to answer the user's question.\n\n"
                "IMPORTANT FOLDER ID CONVENTION:\n"
                "While scripts/data files use normal paths, FOLDER nodes are prefixed in the graph. "
                "To target a folder, you MUST prepend 'folder::scripts::' (or 'folder::data::'). "
                "Example: to inspect contents of '/1. utilities/Modules', use node_id='folder::scripts::/1. utilities/Modules'.\n\n"
                "Typical graph_tool patterns:\n"
                "  • Which scripts import module X? → search X if needed, then neighbors(node_id=X, direction='incoming', relation_type='imports').\n"
                "  • What modules does script X import? → neighbors(node_id=X, direction='outgoing', relation_type='imports').\n"
                "  • What files does script X read/write? → neighbors with relation_type='reads' or 'writes'.\n"
                "  • Global overview of imports/reads/writes → edges(relation_type='imports'|'reads'|'writes')."
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": ["tree", "node", "search", "neighbors", "edges"],
                    "description": "The graph operation to perform. See the MANDATORY WORKFLOW for usage order."
                },
                "path": {
                    "type": "string",
                    "description": "[tree] Folder path to inspect. Default is '/'."
                },
                "domain": {
                    "type": "string",
                    "enum": ["scripts", "data", "both"],
                    "description": "[tree] Graph domain to explore. Start with 'scripts'."
                },
                "max_depth": {
                    "type": "integer",
                    "description": "[tree] Max recursion depth. Use 1 or 2 for a quick overview."
                },
                "node_id": {
                    "type": "string",
                    "description": "[node, neighbors] The exact logical path of the node (e.g., '/Clean/Orders.ion')."
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing", "all", "siblings"],
                    "description": "[neighbors] Think from perspective of node_id: 'incoming' = what targets this node; 'outgoing' = what this node targets."
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["reads", "writes", "imports", "defines", "contains", "sibling"],
                    "description": "[neighbors, edges] Filter by edge type. E.g., 'imports' to find module dependencies."
                },
                "query": {
                    "type": "string",
                    "description": "[search] Node name or path to search. NEVER use this for verbs like 'import' or 'read'."
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["script", "data_file", "table", "function", "folder"]},
                    "description": "[search] Filter by node type."
                },
            },
            required=["action"],
        )
