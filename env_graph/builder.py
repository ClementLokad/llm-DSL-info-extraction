"""
Network Builder Module
======================

Constructs a hierarchical dependency graph from Envision scripts (.nvn files).
The graph includes:
- TWO folder hierarchies: SCRIPTS domain and DATA domain
- Scripts, data files, tables, and functions as nodes
- Relationships: contains, reads, writes, imports, defines, sibling

Architecture:
    SCRIPTS domain: /1. utilities/, /2. Data sanity/, etc. (contains scripts)
    DATA domain: /Input/, /Clean/, /Manual/, etc. (contains data_files)

This module is maintained inside the envision/ repository and uses the
project's local Python environment and configuration layout.
"""

import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import fnmatch

from .typedefs import Network, Node, Edge, NodeType, EdgeType, TreeDomain
from .utils import ConfigLoader
from .extractor import SymbolExtractor

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """
    Builds a hierarchical dependency network from Envision scripts.
    
    The builder performs:
    1. Script parsing and node creation
    2. Dependency extraction (reads, writes, imports)
    3. Internal symbol extraction (tables, functions)
    4. Folder hierarchy construction
    5. Sibling relationship establishment
    
    Attributes:
        network: The constructed Network object
        config: Configuration dictionary
        root_dir: Root directory for script scanning
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the NetworkBuilder.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.parsing_config = self.config.get("parsing", {})
        self.output_config = self.config.get("output", {})
        
        self.root_dir = Path(self.parsing_config.get("script_dir", "scripts"))
        self.script_ext = self.parsing_config.get("script_ext", "nvn")
        self.max_recursion = self.parsing_config.get("recursion_limit", 10)
        self.normalize_brackets = self.parsing_config.get("normalize_brackets", True)
        self.data_extensions = self.parsing_config.get("data_extensions", ["ion"])
        
        self.network = Network()
        self.file_mapping = ConfigLoader.load_mapping(self.config)
        # Create Reverse Mapping (Logical Path -> File ID)
        self.reverse_mapping = {v: k for k, v in self.file_mapping.items()}
        
        # Resolution Tracking
        self.info_resolutions = {
            "globs": [],        # List of {pattern: str, matches: List[str], count: int}
            "placeholders": []  # List of {original: str, resolved: str, source: str}
        }
        
        # Regex Patterns
        flags = re.MULTILINE | re.IGNORECASE
        self.read_pattern = re.compile(r'read\s+["\']([^"\']+)["\']', flags)
        self.write_pattern = re.compile(r'write\s+\w+\s+as\s+["\']([^"\']+)["\']', flags)
        self.show_write_pattern = re.compile(r'write\s*:\s*["\']([^"\']+)["\']', flags)
        self.table_pattern = re.compile(r'table\s+(\w+)\s*=', flags)
        self.import_pattern = re.compile(r'import\s+["\']([^"\']+)["\']', flags)
        
        # Placeholder pattern for variable interpolation
        self.placeholder_pattern = re.compile(r'\\?\{([A-Za-z0-9_]+)\}')
        
        # Bracket normalization pattern: [1], [ 1], [1 ], [ 1 ] -> [ X ]
        self.bracket_pattern = re.compile(r'\[\s*(\d+)\s*\]')
        
        # Function Pattern: Capture everything until '(', '=', or '{'
        self.func_pattern = re.compile(r'(?:process|def)\s+([^{=(]+)', re.IGNORECASE)
        
        # Variable Collection Patterns (for placeholder resolution only)
        self.const_decl_pattern = re.compile(r'const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"')
        self.var_decl_pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"(.*)"')

    # =========================================================================
    # Path Normalization
    # =========================================================================
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path for consistent graph representation.
        
        Performs:
        1. Bracket normalization: [1], [ 1], [1 ] -> [ X ]
        2. Slash normalization: \\ -> /
        3. Leading slash addition for absolute-looking paths
        
        Args:
            path: Raw path string from script
            
        Returns:
            Normalized path string
            
        Examples:
            >>> _normalize_path("/supplier/grid[1]/data.ion")
            '/supplier/grid[ X ]/data.ion'
            >>> _normalize_path("/project/output[ 42]/file.ion")
            '/project/output[ X ]/file.ion'
        """
        result = path
        
        # Normalize brackets: [N] -> [ X ] (with spaces)
        if self.normalize_brackets:
            result = self.bracket_pattern.sub('[ X ]', result)
        
        # Normalize slashes
        result = result.replace('\\', '/')
        
        # Collapse multiple slashes
        while '//' in result:
            result = result.replace('//', '/')
        
        # Add leading slash for paths that look absolute
        if not result.startswith('/') and ('/' in result or '.' in result):
            result = '/' + result
            
        return result
    
    def _extract_execution_order(self, name: str) -> Optional[int]:
        """
        Extract execution order from name pattern.
        
        Supports patterns like:
        - "2. Data sanity" → 2
        - "3 - High Level Data Health" → 3
        - "01 - Catalog" → 1
        - "[ 1 ] - Technical" → 1
        - "0 General Organization" → 0
        
        Args:
            name: Display name of script or folder
            
        Returns:
            Execution order as integer, or None if no numeric prefix found
            
        Examples:
            >>> _extract_execution_order("01 - Catalog and Orders")
            1
            >>> _extract_execution_order("3 - High Level Data Health")
            3
            >>> _extract_execution_order("[ 2 ] - Business Preprocessing")
            2
        """
        if not name:
            return None
        
        # Pattern: "[ X ]" at start (brackets with number)
        match = re.match(r'^\[\s*(\d+)\s*\]', name)
        if match:
            return int(match.group(1))
        
        # Pattern: number followed by separator (., -, _, space)
        # Matches: "2. xxx", "3 - xxx", "01_xxx", "0 xxx"
        match = re.match(r'^(\d+)(?:[.\-_\s]|$)', name)
        if match:
            return int(match.group(1))
        
        return None

    # =========================================================================
    # Content Collection (for placeholder resolution)
    # =========================================================================

    def _collect_constants(self, content: str, script_id: str = "unknown") -> Dict[str, Tuple[str, int]]:
        """
        Collect const and string variable definitions for placeholder resolution.
        
        Parses the script content for variable definitions that might be used
        in path interpolation (e.g., const BasePath = "/project/data").
        
        Args:
            content: Script content (preferably stripped of comments)
            script_id: Identifier for tracking resolution sources
            
        Returns:
            Dictionary mapping variable names to (resolved_value, line_number) tuples
        """
        consts = {}
        
        for idx, line in enumerate(content.splitlines()):
            line = line.strip()
            if not line:
                continue
             
            # Check for 'const' declaration
            match = self.const_decl_pattern.match(line)
            if not match:
                # Check for simple string assignment
                match = self.var_decl_pattern.match(line)
                 
            if match:
                key, value = match.group(1), match.group(2)
                 
                # Resolve using currently known values (top-down)
                current_values = {k: v[0] for k, v in consts.items()}
                resolved_val = self._resolve_placeholders(value, current_values)
                 
                # Track resolution if changed
                if value != resolved_val:
                    self.info_resolutions["placeholders"].append({
                        "original": value,
                        "resolved": resolved_val,
                        "source": f"{script_id}::var::{key}"
                    })
                 
                consts[key] = (resolved_val, idx + 1)
                 
        return consts

    def _resolve_placeholders(self, text: str, consts: Dict[str, str], depth: int = 0) -> str:
        """
        Recursively resolve {placeholder} patterns in text.
        
        Args:
            text: Text containing {placeholder} patterns
            consts: Dictionary of known constant values
            depth: Current recursion depth (prevents infinite loops)
            
        Returns:
            Text with placeholders replaced by their values
        """
        if depth > self.max_recursion:
            return text
        # If not found, keep the original placeholder
        replaced = self.placeholder_pattern.sub(
            lambda match: consts.get(match.group(1), match.group(0)), text
        )
        if replaced == text:
            return replaced
        return self._resolve_placeholders(replaced, consts, depth=depth + 1)

    # =========================================================================
    # Content Extraction Helpers
    # =========================================================================

    def _strip_comments(self, content: str) -> str:
        """
        Strip comments while preserving string literals.
        
        Handles:
        - Block comments: /* ... */
        - Line comments: //
        
        Args:
            content: Raw script content
            
        Returns:
            Content with comments replaced by spaces
        """
        pattern = re.compile(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(/\*[^*]*\*+(?:[^/*][^*]*\*+)*/|//[^\n]*)',
            re.MULTILINE | re.DOTALL
        )
        return pattern.sub(lambda m: m.group(1) if m.group(1) else " ", content)

    def _extract_docs(self, content: str) -> Dict[str, List[str]]:
        """
        Extract structured documentation comments from script content.
        
        Envision documentation conventions:
        - /// : Structural documentation (code structure)
        - //' : Business documentation (domain logic)
        - \"\"\" : User-facing markdown blocks
        - //// : Internal memos
        
        Args:
            content: Raw script content
            
        Returns:
            Dictionary with keys: structure, business, user, memos
            Each value is a list of extracted documentation strings
        """
        docs = {
            "structure": [],  # ///
            "business": [],   # //'
            "user": [],       # """
            "memos": []       # ////
        }
        
        lines = content.splitlines()
        in_markdown = False
        markdown_buf = []
        
        for line in lines:
            stripped = line.strip()
            
            # Markdown block toggle
            if stripped.startswith('"""'):
                if in_markdown:
                    # End block
                    markdown_buf.append(stripped.replace('"""', ''))
                    docs["user"].append("\n".join(markdown_buf))
                    markdown_buf = []
                    in_markdown = False
                else:
                    # Start block
                    in_markdown = True
                    markdown_buf.append(stripped.replace('"""', ''))
                continue
            
            if in_markdown:
                markdown_buf.append(line)  # Keep indentation
                continue

            # Line comments
            if stripped.startswith("////"):
                docs["memos"].append(stripped[4:].strip())
            elif stripped.startswith("///"):
                docs["structure"].append(stripped[3:].strip())
            elif stripped.startswith("//'"):
                docs["business"].append(stripped[3:].strip())
                
        return docs

    def _extract_function_body(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """
        Extract the full body of a function definition.
        
        Uses indentation-based block detection for Envision's structure.
        
        Args:
            lines: List of all lines in the script
            start_idx: Index of the function definition line
            
        Returns:
            Tuple of (body_content, end_line_number_1_indexed)
        """
        body = [lines[start_idx]]  # Include definition line
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        last_idx = start_idx
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():  # Empty lines are part of body
                body.append(line)
                last_idx = i
                continue
                
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent > base_indent:
                body.append(line)
                last_idx = i
            else:
                break  # End of block
                
        return "\n".join(body), last_idx + 1

    # =========================================================================
    # Main Build Process
    # =========================================================================

    def build(self) -> Network:
        """
        Build the complete dependency network.
        
        Process:
        1. Scan and parse all script files
        2. Extract dependencies and internal symbols
        3. Resolve glob patterns to concrete files
        4. Build folder hierarchy
        5. Establish sibling relationships
        6. Save network and metadata
        
        Returns:
            The constructed Network object
        """
        files = list(self.root_dir.glob(f"*.{self.script_ext}"))
        print(f"🔍 NetworkBuilder: Scanning {len(files)} files in {self.root_dir}...")
        
        # Phase 1: Process all script files
        for file_path in files:
            self._process_file(file_path)

        # Phase 2: Resolve glob patterns
        self._resolve_glob_nodes()

        # Phase 3: Build folder hierarchy
        self._build_folder_hierarchy()
        
        # Phase 4: Build sibling relationships
        self._build_sibling_edges()

        # Report placeholder resolutions
        cascade_count = len(self.info_resolutions["placeholders"])
        if cascade_count > 0:
            print(f"🔄 Resolved {cascade_count} placeholder cascades.")

        # Save outputs
        self._save_network()
        self._save_metadata(len(files))
        
        return self.network

    def _resolve_glob_nodes(self):
        """
        Resolve glob pattern nodes to concrete file nodes.
        
        Identifies nodes with '*' patterns, finds matching concrete nodes,
        redirects all edges, and removes the glob node.
        """
        all_node_ids = list(self.network.nodes.keys())
        glob_nodes = [nid for nid in all_node_ids if '*' in nid]
        
        # Match only against DATA_FILE nodes
        candidate_nodes = [
            nid for nid in all_node_ids 
            if '*' not in nid and self.network.nodes[nid].type == NodeType.DATA_FILE
        ]
        
        resolved_count = 0
        
        for glob_id in glob_nodes:
            matches = [cand for cand in candidate_nodes if fnmatch.fnmatch(cand, glob_id)]
            
            if not matches:
                continue
            
            # Record stats
            self.info_resolutions["globs"].append({
                "pattern": glob_id,
                "matches": matches,
                "count": len(matches)
            })
                
            # Redirect edges
            edges_to_remove = []
            edges_to_add = []
            
            for edge in self.network.edges:
                if edge.target == glob_id:
                    edges_to_remove.append(edge)
                    for match_id in matches:
                        new_meta = edge.metadata.copy() if edge.metadata else {}
                        new_meta["glob_source"] = glob_id
                        
                        edges_to_add.append(Edge(
                            source=edge.source,
                            target=match_id,
                            type=edge.type,
                            metadata=new_meta
                        ))
            
            if edges_to_add:
                # Apply changes
                for e in edges_to_remove:
                    self.network.remove_edge(e)
                for e in edges_to_add:
                    self.network.add_edge(e)
                
                # Remove the original glob node
                self.network.nodes.pop(glob_id, None)
                resolved_count += 1
                
        if resolved_count > 0:
            print(f"✨ Resolved {resolved_count} glob patterns to concrete files.")

    def _build_folder_hierarchy(self):
        """
        Build TWO separate folder hierarchies: SCRIPTS and DATA domains.
        
        Creates FOLDER nodes for each directory level in the logical paths
        and establishes CONTAINS edges from folders to their children.
        
        SCRIPTS domain: folders derived from script paths
            - folder ID: "folder::scripts::/1. utilities"
            - Contains only script files
            
        DATA domain: folders derived from data_file paths
            - folder ID: "folder::data::/Clean/tmp"
            - Contains only data files
            
        Both domains have their own root folder (/).
        """
        # Separate folder paths by domain
        scripts_folders: Set[str] = set()  # From script paths
        data_folders: Set[str] = set()     # From data_file paths
        
        # Track which nodes belong to which folder (with domain)
        # node_id -> (parent_folder_path, domain)
        node_folders: Dict[str, Tuple[str, TreeDomain]] = {}
        
        for node in self.network.nodes.values():
            if not node.path:
                continue
                
            # Determine domain based on node type
            if node.type == NodeType.SCRIPT:
                domain = TreeDomain.SCRIPTS
                folder_set = scripts_folders
            elif node.type == NodeType.DATA_FILE:
                domain = TreeDomain.DATA
                folder_set = data_folders
            else:
                continue  # Skip folders, tables, functions
                
            # Get the directory part of the path
            path_obj = Path(node.path)
            parent = path_obj.parent.as_posix()
            
            if parent and parent != '.':
                node_folders[node.id] = (parent, domain)
                
                # Add all ancestor folders to this domain
                current = path_obj.parent
                while current.as_posix() not in ('', '.', '/'):
                    folder_set.add(current.as_posix())
                    current = current.parent
                
                # Add root if path is absolute
                if node.path.startswith('/'):
                    folder_set.add('/')
        
        # Create folder nodes for SCRIPTS domain
        for folder_path in scripts_folders:
            folder_id = f"folder::scripts::{folder_path}"
            
            if folder_id not in self.network.nodes:
                folder_short_name = Path(folder_path).name or '/'
                metadata = {"domain": TreeDomain.SCRIPTS.value}
                # Extract execution_order from folder name
                exec_order = self._extract_execution_order(folder_short_name)
                if exec_order is not None:
                    metadata["execution_order"] = exec_order
                # Note: name=folder_path (full path) for easier identification in lite mode
                folder_node = Node(
                    id=folder_id,
                    type=NodeType.FOLDER,
                    name=folder_path,
                    path=folder_path,
                    metadata=metadata
                )
                self.network.add_node(folder_node)
        
        # Create folder nodes for DATA domain
        for folder_path in data_folders:
            folder_id = f"folder::data::{folder_path}"
            
            if folder_id not in self.network.nodes:
                folder_short_name = Path(folder_path).name or '/'
                metadata = {"domain": TreeDomain.DATA.value}
                # Extract execution_order from folder name
                exec_order = self._extract_execution_order(folder_short_name)
                if exec_order is not None:
                    metadata["execution_order"] = exec_order
                # Note: name=folder_path (full path) for easier identification in lite mode
                folder_node = Node(
                    id=folder_id,
                    type=NodeType.FOLDER,
                    name=folder_path,
                    path=folder_path,
                    metadata=metadata
                )
                self.network.add_node(folder_node)
        
        # Create CONTAINS edges: folder -> child folders (SCRIPTS domain)
        for folder_path in scripts_folders:
            folder_id = f"folder::scripts::{folder_path}"
            parent_path = Path(folder_path).parent.as_posix()
            
            if parent_path and parent_path != '.' and parent_path != folder_path:
                parent_id = f"folder::scripts::{parent_path}"
                if parent_id in self.network.nodes:
                    self.network.add_edge(Edge(
                        source=parent_id,
                        target=folder_id,
                        type=EdgeType.CONTAINS
                    ))
        
        # Create CONTAINS edges: folder -> child folders (DATA domain)
        for folder_path in data_folders:
            folder_id = f"folder::data::{folder_path}"
            parent_path = Path(folder_path).parent.as_posix()
            
            if parent_path and parent_path != '.' and parent_path != folder_path:
                parent_id = f"folder::data::{parent_path}"
                if parent_id in self.network.nodes:
                    self.network.add_edge(Edge(
                        source=parent_id,
                        target=folder_id,
                        type=EdgeType.CONTAINS
                    ))
        
        # Create CONTAINS edges: folder -> files (both domains)
        for node_id, (parent_folder, domain) in node_folders.items():
            prefix = "scripts" if domain == TreeDomain.SCRIPTS else "data"
            parent_id = f"folder::{prefix}::{parent_folder}"
            if parent_id in self.network.nodes:
                self.network.add_edge(Edge(
                    source=parent_id,
                    target=node_id,
                    type=EdgeType.CONTAINS
                ))
        
        total_folders = len(scripts_folders) + len(data_folders)
        print(f"📁 Created {total_folders} folder nodes ({len(scripts_folders)} scripts, {len(data_folders)} data).")

    def _build_sibling_edges(self):
        """
        Build sibling relationships between files in the same folder.
        
        CRITICAL: Siblings are INTRA-DOMAIN only:
        - Scripts are siblings with other scripts (same script folder)
        - Data files are siblings with other data files (same data folder)
        - A script is NEVER sibling to a data file
        
        Sibling edges are bidirectional (stored once, interpreted both ways).
        Only connects nodes at the same level in the hierarchy.
        """
        # Group nodes by their parent folder AND node type (for domain separation)
        # Key: (folder_path, domain) -> list of node_ids
        folder_children: Dict[Tuple[str, str], List[str]] = {}
        
        for node in self.network.nodes.values():
            # Skip folders and internal nodes (tables, functions)
            if node.type == NodeType.FOLDER:
                continue
            if '::' in node.id and not node.id.startswith('folder::'):
                continue  # Skip internal nodes like script::table::X
                
            if not node.path:
                continue
                
            # Determine domain from node type
            if node.type == NodeType.SCRIPT:
                domain = "scripts"
            elif node.type == NodeType.DATA_FILE:
                domain = "data"
            else:
                continue
                
            parent_folder = Path(node.path).parent.as_posix()
            if parent_folder and parent_folder != '.':
                key = (parent_folder, domain)
                if key not in folder_children:
                    folder_children[key] = []
                folder_children[key].append(node.id)
        
        # Create sibling edges within each folder (same domain only)
        sibling_count = 0
        scripts_siblings = 0
        data_siblings = 0
        
        for (folder_path, domain), children in folder_children.items():
            if len(children) < 2:
                continue
                
            # Sort for deterministic ordering
            children_sorted = sorted(children)
            
            # Create edges between consecutive siblings (chain pattern)
            for i in range(len(children_sorted) - 1):
                self.network.add_edge(Edge(
                    source=children_sorted[i],
                    target=children_sorted[i + 1],
                    type=EdgeType.SIBLING,
                    metadata={"folder": folder_path, "domain": domain}
                ))
                sibling_count += 1
                if domain == "scripts":
                    scripts_siblings += 1
                else:
                    data_siblings += 1
        
        print(f"🔗 Created {sibling_count} sibling relationships ({scripts_siblings} scripts, {data_siblings} data).")

    # =========================================================================
    # File Processing
    # =========================================================================

    def _process_file(self, file_path: Path):
        """
        Process a single script file and add its nodes/edges to the network.
        
        Creates:
        - SCRIPT node with full content and metadata
        - Dependency edges (READS, WRITES, IMPORTS)
        - Internal nodes (TABLE, FUNCTION) with DEFINES edges
        
        Args:
            file_path: Path to the .nvn script file
        """
        logical_path = ConfigLoader.get_logical_path(file_path.name, self.file_mapping, self.script_ext)
        real_id = file_path.name.replace(f".{self.script_ext}", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return

        # Extract documentation
        docs = self._extract_docs(content)
        
        # Extract execution order from display name (not filename)
        display_name = Path(logical_path).name
        exec_order = self._extract_execution_order(display_name)
        
        # Build metadata
        metadata = {
            "docs": docs,
            "symbols": SymbolExtractor.extract(self._strip_comments(content))
        }
        if exec_order is not None:
            metadata["execution_order"] = exec_order
        
        # Create Script Node with FULL CONTENT
        # Note: name=logical_path (full path) for easier identification in lite mode
        script_node = Node(
            id=real_id, 
            type=NodeType.SCRIPT,
            name=logical_path,
            path=logical_path, 
            content=content, 
            metadata=metadata
        )
        self.network.add_node(script_node)
        
        self._extract_dependencies(content, script_node)

    def _extract_dependencies(self, content: str, script_node: Node):
        """
        Extract all dependencies from script content.
        
        Creates:
        - DATA_FILE nodes for read/write targets
        - SCRIPT nodes for import targets
        - TABLE nodes for table definitions
        - FUNCTION nodes for function definitions
        - Corresponding edges (READS, WRITES, IMPORTS, DEFINES)
        
        Args:
            content: Script content
            script_node: The parent script node
        """
        def get_line_num(match_obj):
            return content[:match_obj.start()].count('\n') + 1

        clean_content = self._strip_comments(content)
        consts_map = self._collect_constants(clean_content, script_node.id) 
        consts_values = {k: v[0] for k, v in consts_map.items()}

        edges_found = {} 
        target_types = {} 

        def process_match(raw: str, edge_type: EdgeType, target_type: NodeType = NodeType.DATA_FILE):
            """Process a regex match and prepare edge creation."""
            # Resolve placeholders
            raw_path = self._resolve_placeholders(raw, consts_values)
            
            # Track resolution
            if raw != raw_path:
                self.info_resolutions["placeholders"].append({
                    "original": raw,
                    "resolved": raw_path,
                    "source": f"{script_node.id} (Edge)"
                })
            
            # Replace leftover interpolation with glob pattern
            if '{' in raw_path and '}' in raw_path:
                raw_path = re.sub(r'\{[^}]+\}', '*', raw_path)
            
            # Normalize the path
            target_path = self._normalize_path(raw_path)
            
            key = (target_path, edge_type)
            if key not in edges_found:
                edges_found[key] = []
            edges_found[key].append(raw)
            target_types[target_path] = target_type

        # 1. READS
        for match in self.read_pattern.finditer(clean_content):
            process_match(match.group(1), EdgeType.READS, NodeType.DATA_FILE)

        # 2. WRITES
        for match in self.write_pattern.finditer(clean_content):
            process_match(match.group(1), EdgeType.WRITES, NodeType.DATA_FILE)
        for match in self.show_write_pattern.finditer(clean_content):
            process_match(match.group(1), EdgeType.WRITES, NodeType.DATA_FILE)

        # 3. IMPORTS
        for match in self.import_pattern.finditer(clean_content):
            raw = match.group(1)
            raw_path = self._resolve_placeholders(raw, consts_values)
            
            if raw != raw_path:
                self.info_resolutions["placeholders"].append({
                    "original": raw,
                    "resolved": raw_path,
                    "source": f"{script_node.id} (Import)"
                })
            
            clean_path = self._normalize_path(raw_path)
            
            # Try to resolve to known script
            target_id = self.reverse_mapping.get(clean_path)
            if not target_id:
                target_id = self.reverse_mapping.get(clean_path.lstrip('/'))
            
            if target_id:
                key = (target_id, EdgeType.IMPORTS)
                if key not in edges_found:
                    edges_found[key] = []
                edges_found[key].append(raw)
                target_types[target_id] = NodeType.SCRIPT
            else:
                key = (clean_path, EdgeType.IMPORTS)
                if key not in edges_found:
                    edges_found[key] = []
                edges_found[key].append(raw)
                target_types[clean_path] = NodeType.SCRIPT

        # Create edges and target nodes
        for (target_id, edge_type), occurrences in edges_found.items():
            if target_id not in self.network.nodes:
                name = Path(target_id).name if '/' in target_id else target_id
                t_node = Node(
                    id=target_id, 
                    type=target_types[target_id], 
                    name=name,
                    path=target_id if '/' in target_id else None
                )
                self.network.add_node(t_node)
            
            meta = {
                "count": len(occurrences),
                "occurrences": occurrences,
            }
            self.network.add_edge(Edge(
                source=script_node.id, 
                target=target_id, 
                type=edge_type, 
                metadata=meta
            ))

        # 4. TABLES - Extract definition line for content
        lines = content.splitlines()
        for match in self.table_pattern.finditer(clean_content):
            name = match.group(1)
            node_id = f"{script_node.id}::table::{name}"
            lineno = get_line_num(match)
            
            # Get the table definition line as content
            table_def_line = lines[lineno - 1] if lineno <= len(lines) else ""
            
            table_node = Node(
                id=node_id, 
                type=NodeType.TABLE, 
                name=name, 
                content=table_def_line.strip(),
                start_line=lineno, 
                end_line=lineno
            )
            self.network.add_node(table_node)
            self.network.add_edge(Edge(
                source=script_node.id, 
                target=node_id, 
                type=EdgeType.DEFINES
            ))
            
        # 5. FUNCTIONS
        for idx, line in enumerate(lines):
            match = self.func_pattern.search(line)
            if match and not line.strip().startswith("//"):
                full_match = match.group(1).strip()
                parts = full_match.split()
                if not parts:
                    continue
                
                func_name = parts[-1]
                qualifiers = parts[:-1]
                
                node_id = f"{script_node.id}::func::{func_name}"
                body, end_lineno = self._extract_function_body(lines, idx)
                start_lineno = idx + 1
                
                meta = {}
                if qualifiers:
                    meta["qualifiers"] = qualifiers
                
                func_node = Node(
                    id=node_id, 
                    type=NodeType.FUNCTION, 
                    name=func_name, 
                    content=body, 
                    start_line=start_lineno,
                    end_line=end_lineno,
                    metadata=meta
                )
                self.network.add_node(func_node)
                self.network.add_edge(Edge(
                    source=script_node.id, 
                    target=node_id, 
                    type=EdgeType.DEFINES
                ))

    # =========================================================================
    # Output Generation
    # =========================================================================

    def _save_network(self):
        """Save the network to JSON file."""
        out_path = Path(self.output_config["network_file"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.network.to_dict(), f, indent=2)
        print(f"✅ Network saved to {out_path}")

    def _save_metadata(self, file_count: int):
        """
        Save network metadata and statistics.
        
        Includes:
        - Generation timestamp
        - Node and edge counts by type
        - Resolution statistics (globs, placeholders)
        
        Args:
            file_count: Number of source files processed
        """
        out_path = Path(self.output_config["metadata_file"])
        
        stats = {
            "generated_at": datetime.now().isoformat(),
            "source_files": file_count,
            "node_count": len(self.network.nodes),
            "edge_count": len(self.network.edges),
            "nodes_by_type": {},
            "edges_by_type": {},
            "resolutions": self.info_resolutions
        }
        
        for n in self.network.nodes.values():
            t = n.type.value
            stats["nodes_by_type"][t] = stats["nodes_by_type"].get(t, 0) + 1
            
        for e in self.network.edges:
            t = e.type.value
            stats["edges_by_type"][t] = stats["edges_by_type"].get(t, 0) + 1
            
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Metadata saved to {out_path}")
