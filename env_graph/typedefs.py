"""
Type Definitions for the Envision Dependency Graph.

This module defines the core data structures used throughout the graph system:
- TreeDomain: Enumeration of folder hierarchy domains (scripts vs data)
- NodeType: Enumeration of all possible node types in the graph
- EdgeType: Enumeration of all possible edge/relation types
- Node: Data class representing a single node in the graph
- Edge: Data class representing a relationship between nodes
- Network: Container class holding all nodes and edges

Architecture Overview:
    
    TWO SEPARATE FOLDER TREES:
    ==========================
    
    SCRIPTS domain:           DATA domain:
    ├── /1. utilities         ├── /Input
    │   ├── /1. populating    │   ├── /Catalog
    │   └── /Modules          │   └── /Orders
    ├── /2. Data sanity       ├── /Clean
    └── /3. Inspectors            ├── /tmp
                                  └── /Override
                              └── /Manual
    
    
    NODES HIERARCHY:
    ================
    
    folder          Directories (belong to either SCRIPTS or DATA domain)
        │
        ├── script      Envision code files (.nvn) - HAS content
        │                   → Only in SCRIPTS domain
        │
        └── data_file   Data files (.ion) - NO content (black boxes)
                            → Only in DATA domain
    
    script
        │
        ├── table       Table definitions within a script
        │
        └── function    Function definitions within a script
    
    
    EDGES (RELATIONS):
    ==================
    
    Intra-domain:
    - contains:  folder → folder/script/data_file  (hierarchy, same domain)
    - sibling:   file ←→ file                      (same parent, same domain)
    - imports:   script → script                   (SCRIPTS domain only)
    
    Cross-domain:
    - reads:     script → data_file                (SCRIPTS → DATA)
    - writes:    script → data_file                (SCRIPTS → DATA)
    
    Internal:
    - defines:   script → table/function           (symbol definition)

"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class TreeDomain(str, Enum):
    """
    Domain for folder hierarchies - separates scripts from data storage.
    
    The Envision project has two distinct directory trees:
    
    SCRIPTS Domain:
        Contains script files (.nvn) organized by workflow:
        - /1. utilities/
        - /2. Data sanity/
        - /3. Inspectors/
        - etc.
    
    DATA Domain:
        Contains data files (inputs/outputs) in storage paths:
        - /Input/     (raw data inputs)
        - /Clean/     (processed data)
        - /Manual/    (manual overrides)
        - /Background/ (reference data)
        
    Relationships:
        - siblings are INTRA-domain (scripts with scripts, data with data)
        - imports are INTRA-domain (script → script only)
        - reads/writes are CROSS-domain (script → data_file)
    """
    SCRIPTS = "scripts"   # Folder tree for .nvn script files
    DATA = "data"         # Folder tree for data files (ion, csv, xlsx...)


class NodeType(str, Enum):
    """
    Enumeration of all node types in the Envision graph.
    
    Types:
        FOLDER:     Directory node (deduced from paths of scripts/data_files)
                    - Has NO content
                    - Example: "/1. utilities/Modules"
        
        SCRIPT:     Envision code file (.nvn extension, configurable)
                    - HAS content (full source code)
                    - Example: "68010" with path "/3. Inspectors/2 - Sales Analysis"
        
        DATA_FILE:  Data file (.ion extension, configurable)
                    - Has NO content (treated as black box)
                    - Example: "/Clean/Items.ion"
        
        TABLE:      Table definition within a script
                    - HAS content (the definition line)
                    - ID format: "{script_id}::table::{TableName}"
                    - Example: "68010::table::ItemsWeek"
        
        FUNCTION:   Function definition within a script
                    - HAS content (full function body)
                    - ID format: "{script_id}::func::{FunctionName}"
                    - Example: "67992::func::StockEvol"
    """
    FOLDER = "folder"
    SCRIPT = "script"
    DATA_FILE = "data_file"
    TABLE = "table"
    FUNCTION = "function"


class EdgeType(str, Enum):
    """
    Enumeration of all edge/relation types in the Envision graph.
    
    Directed Arcs (source → target):
        CONTAINS:   Folder contains its children (hierarchy)
                    - source: folder
                    - target: folder | script | data_file
        
        READS:      Script reads data from a file
                    - source: script
                    - target: data_file
        
        WRITES:     Script writes data to a file
                    - source: script
                    - target: data_file
        
        IMPORTS:    Script imports another script as module
                    - source: script
                    - target: script
        
        DEFINES:    Script defines a symbol (table or function)
                    - source: script
                    - target: table | function
    
    Undirected Edge (bidirectional):
        SIBLING:    Two files share the same parent folder
                    - Both nodes are script | data_file
                    - Stored once (not duplicated as A→B and B→A)
    """
    # Directed arcs
    CONTAINS = "contains"
    READS = "reads"
    WRITES = "writes"
    IMPORTS = "imports"
    DEFINES = "defines"
    
    # Undirected edge
    SIBLING = "sibling"


@dataclass
class Node:
    """
    Represents a single node in the Envision dependency graph.
    
    Attributes:
        id (str): 
            Unique identifier for the node.
            - For folders: logical path (e.g., "/1. utilities/Modules")
            - For scripts: file ID from mapping (e.g., "68010")
            - For data_files: logical path (e.g., "/Clean/Items.ion")
            - For tables: "{script_id}::table::{name}" (e.g., "68010::table::ItemsWeek")
            - For functions: "{script_id}::func::{name}" (e.g., "67992::func::StockEvol")
        
        type (NodeType): 
            The type of this node (folder, script, data_file, table, function)
        
        name (str | None): 
            Human-readable short name.
            - For folders: folder name (e.g., "Modules")
            - For scripts: script name (e.g., "2 - Sales Analysis")
            - For data_files: filename (e.g., "Items.ion")
            - For tables/functions: symbol name (e.g., "ItemsWeek", "StockEvol")
        
        path (str | None): 
            Full logical path in the file system.
            - For folders: "/1. utilities/Modules"
            - For scripts: "/3. Inspectors/2 - Sales Analysis"
            - For data_files: "/Clean/Items.ion"
            - For tables/functions: None (use parent_script_path in metadata)
        
        content (str | None): 
            The actual content/source code.
            - For scripts: full source code
            - For functions: function body
            - For tables: definition line (e.g., "table ItemsWeek = cross(Items, Week)")
            - For folders/data_files: None (no content)
        
        start_line (int | None): 
            1-indexed line number where this symbol starts.
            - Only relevant for tables and functions
        
        end_line (int | None): 
            1-indexed line number where this symbol ends.
            - Only relevant for tables and functions
        
        metadata (Dict[str, Any]): 
            Additional structured information. Contents vary by node type:
            
            For FOLDER:
                {
                    "execution_order": int | None,  # From [ X ] pattern in name
                    "depth": int,                   # Depth in tree (root=0)
                    "parent_folder": str,           # Parent folder path
                    "children_count": int           # Number of direct children
                }
            
            For SCRIPT:
                {
                    "execution_order": int | None,  # From name pattern
                    "parent_folder": str,           # Parent folder path
                    "lines_count": int,             # Total lines in file
                    "docs": {
                        "structure": [...],         # /// comments
                        "business": [...],          # //' comments
                        "user": [...],              # triple-quote blocks
                        "memos": [...]              # //// comments
                    },
                    "symbols": {
                        "tables": {"TableName": count, ...},
                        "functions": {"FuncName": count, ...}
                    }
                }
            
            For DATA_FILE:
                {
                    "parent_folder": str,           # Parent folder path
                    "extension": str                # File extension (e.g., "ion")
                }
            
            For TABLE:
                {
                    "parent_script": str,           # Script ID that defines this table
                    "parent_script_path": str       # Script's logical path
                }
            
            For FUNCTION:
                {
                    "parent_script": str,           # Script ID
                    "parent_script_path": str,      # Script's logical path
                    "qualifiers": [...],            # e.g., ["process", "pure"]
                    "lines_count": int              # Number of lines in function
                }
    
    Example (SCRIPT):
        Node(
            id="68010",
            type=NodeType.SCRIPT,
            name="2 - Sales Analysis",
            path="/3. Inspectors/2 - Sales Analysis",
            content="/// input: /clean/\\nimport ...",
            start_line=None,
            end_line=None,
            metadata={
                "execution_order": 2,
                "parent_folder": "/3. Inspectors",
                "lines_count": 234,
                "docs": {"structure": [...], "business": [], "user": [...], "memos": []},
                "symbols": {"tables": {"ItemsWeek": 1}, "functions": {}}
            }
        )
    
    Example (FUNCTION):
        Node(
            id="67992::func::StockEvol",
            type=NodeType.FUNCTION,
            name="StockEvol",
            path=None,
            content="def process StockEvol(horizon: number)\\n  ...",
            start_line=45,
            end_line=78,
            metadata={
                "parent_script": "67992",
                "parent_script_path": "/1. utilities/Modules/Functions",
                "qualifiers": ["process"],
                "lines_count": 34
            }
        )
    """
    id: str
    type: NodeType
    name: Optional[str] = None
    path: Optional[str] = None
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the node to a dictionary.
        
        Returns:
            Dict with all node attributes. The 'type' field is converted
            to its string value for JSON compatibility.
        
        Example Output:
            {
                "id": "68010",
                "type": "script",
                "name": "2 - Sales Analysis",
                "path": "/3. Inspectors/2 - Sales Analysis",
                "content": "/// input: /clean/...",
                "start_line": null,
                "end_line": null,
                "metadata": {...}
            }
        """
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, NodeType) else self.type,
            "name": self.name,
            "path": self.path,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata
        }


@dataclass
class Edge:
    """
    Represents a relationship (edge) between two nodes in the graph.
    
    Attributes:
        source (str): 
            ID of the source node (for directed arcs) or first node (for undirected edges).
        
        target (str): 
            ID of the target node (for directed arcs) or second node (for undirected edges).
        
        type (EdgeType): 
            The type of relationship (contains, reads, writes, imports, defines, sibling).
        
        metadata (Dict[str, Any]): 
            Additional information about the edge:
            
            For READS/WRITES:
                {
                    "count": int,           # Number of occurrences in the script
                    "occurrences": [...]    # Raw path strings as found in code
                }
            
            For IMPORTS:
                {
                    "count": int,
                    "occurrences": [...]
                }
            
            For CONTAINS:
                {}  # Usually empty
            
            For DEFINES:
                {}  # Usually empty
            
            For SIBLING:
                {
                    "parent_folder": str    # The shared parent folder
                }
    
    Note on SIBLING edges:
        Sibling edges are UNDIRECTED. They are stored only once (not as both A→B and B→A).
        When querying neighbors with direction="siblings", the API checks both source and
        target fields to find all siblings.
    
    Example (READS):
        Edge(
            source="68010",
            target="/Clean/Items.ion",
            type=EdgeType.READS,
            metadata={"count": 1, "occurrences": ["\\{inputFolder}Items.ion"]}
        )
    
    Example (SIBLING):
        Edge(
            source="68009",
            target="68010",
            type=EdgeType.SIBLING,
            metadata={"parent_folder": "/3. Inspectors"}
        )
    """
    source: str
    target: str
    type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the edge to a dictionary.
        
        Returns:
            Dict with all edge attributes. The 'type' field is converted
            to its string value for JSON compatibility.
        
        Example Output:
            {
                "source": "68010",
                "target": "/Clean/Items.ion",
                "type": "reads",
                "metadata": {"count": 1, "occurrences": [...]}
            }
        """
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value if isinstance(self.type, EdgeType) else self.type,
            "metadata": self.metadata
        }


@dataclass
class Network:
    """
    Container class holding the complete Envision dependency graph.
    
    Attributes:
        nodes (Dict[str, Node]): 
            Dictionary mapping node IDs to Node objects.
            Example: {"68010": Node(...), "/Clean/Items.ion": Node(...)}
        
        edges (List[Edge]): 
            List of all edges (relationships) in the graph.
    
    Methods:
        add_node(node):     Add a node to the graph
        add_edge(edge):     Add an edge to the graph
        remove_edge(edge):  Remove an edge from the graph
        to_dict():          Serialize to dictionary for JSON export
    
    Example Usage:
        network = Network()
        network.add_node(Node(id="68010", type=NodeType.SCRIPT, ...))
        network.add_node(Node(id="/Clean/Items.ion", type=NodeType.DATA_FILE, ...))
        network.add_edge(Edge(source="68010", target="/Clean/Items.ion", type=EdgeType.READS))
    
    Serialized Output (to_dict):
        {
            "nodes": {
                "68010": {"id": "68010", "type": "script", ...},
                "/Clean/Items.ion": {"id": "/Clean/Items.ion", "type": "data_file", ...}
            },
            "edges": [
                {"source": "68010", "target": "/Clean/Items.ion", "type": "reads", ...}
            ]
        }
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the graph. If a node with the same ID exists, it is overwritten.
        
        Args:
            node: The Node object to add.
        """
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the graph.
        
        Args:
            edge: The Edge object to add.
        
        Note:
            Does not check for duplicates. Caller should ensure uniqueness if needed.
        """
        self.edges.append(edge)

    def remove_edge(self, edge: Edge) -> None:
        """
        Removes an edge from the graph if it exists.
        
        Args:
            edge: The Edge object to remove.
        """
        if edge in self.edges:
            self.edges.remove(edge)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the entire network to a dictionary suitable for JSON export.
        
        Returns:
            {
                "nodes": {node_id: node.to_dict(), ...},
                "edges": [edge.to_dict(), ...]
            }
        """
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges]
        }
