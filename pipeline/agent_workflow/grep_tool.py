from importlib import metadata
import re
import pickle
import os
from typing import List, Optional, Dict, Any, Tuple

from rag.core.base_retriever import RetrievalResult, CodeChunk
from rag.core.base_parser import CodeBlock, BlockType 
from rag.parsers.envision_parser import EnvisionParser
from pipeline.agent_workflow.workflow_base import BaseGrepTool, _tool_desc
from get_mapping import get_file_mapping
from config_manager import get_config
from rag.utils.script_scanner import collect_constants, scan_string_for_references, replace_constants_in_script


class GrepTool(BaseGrepTool):
    """
    A retriever that ignores embeddings and performs pure grep-like
    regex-based matching on stored CodeChunks.
    """

    def __init__(self, search_dirs: Optional[List[str]] = None):
        self.config = get_config()
        if search_dirs is None:
            search_dirs = self.config.get('paths.input_dirs', [])
        super().__init__(search_dirs)
        self._embedding_dim = None
        self._blocks: List[CodeBlock] = []
        self.mapping = get_file_mapping()
        self.index_path = self.config.get('main_pipeline.grep_tool.index_path')
        self.case_sensitive = self.config.get('main_pipeline.grep_tool.case_sensitive', False)
        try:
            self.load_index()
        except:
            self.build_save_index()
    
    def check_suffix_match(self, f: str, L: List[str], inverse = False) -> bool:
        """
        Checks if any path 'g' in L (without extension) is a suffix of 
        path 'f' (without extension).

        Args:
            f (str): The main file path.
            L (list): A list of potentially incomplete file paths.
            inverse (bool): If True, checks if f is a suffix of any path in L.

        Returns:
            bool: True if a suffix match is found, False otherwise.
        """
        
        def strip_extension(file_path):
            """Helper to safely remove the file extension from a path."""
            # os.path.splitext returns a tuple (root, ext)
            root, ext = os.path.splitext(file_path)
            return root

        # 1. Strip the extension from the main file path f
        # This gives us the full path/filename without the final .ext
        f_root = strip_extension(f)
        
        # 2. Iterate through the list L and perform the check
        for g in L:
            # Strip the extension from the path in the list
            g_root = strip_extension(g)
            
            # Check if g_root is a suffix of f_root
            # The 'endswith()' string method performs this check efficiently.
            if inverse:
                if g_root.endswith(f_root):
                    return True # Match found, exit immediately
            else:
                if f_root.endswith(g_root):
                    return True # Match found, exit immediately

        # 3. If the loop completes without finding a match, return False
        return False
    

        

    def search(
        self,
        pattern: str,
        source_regex: Optional[str] = None,
        bloc_type: List[BlockType] = None,
        warnings: Optional[List[str]] = []
    ) -> List[RetrievalResult]:
        """Returns a list of RetrievalResult objects where the pattern matches the content of the CodeChunk.

        Args:
            pattern (str): A regex pattern to search for in the content of the CodeChunks.
            source_regex (Optional[str], optional): A regex describing the source files to search in. Defaults to None.
            bloc_type (List[BlockType], optional): A list of block types to filter by. Defaults to None.
            warnings: A list of string that is updated if sources_regex is invalid
        Returns:
            List[RetrievalResult]: A list of RetrievalResult objects containing the matching CodeChunks and their metadata.
        """
        
        if source_regex is not None:
            source_regex = source_regex.strip().strip("'\"")
            valid = False
            for f in self.mapping.values():
                if re.search(source_regex, f, 0 if self.case_sensitive else re.IGNORECASE):
                    valid = True
                    break
            if not valid:
                print(f"Warning: No files in the mapping match the source_regex '{source_regex}'. Ignoring source filter.")
                warnings.append("No files in the codebase match the source_regex so source filter was ignored.")
                source_regex = None
            
        path_regex = r"(\/|\\)|(\.[a-zA-Z0-9]+$)"
        is_path_search = bool(re.search(path_regex, pattern))
        
        file_consts = {}
        
        clean_target = pattern.strip().strip("'\"").strip("/")
        flags = 0 if self.case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(clean_target, flags)
        except re.error:
            regex = re.compile(re.escape(clean_target), flags)

        matches = []

        for block in self._blocks:
            # If block type is specified, filter by it
            if bloc_type is not None and block.block_type not in bloc_type:
                continue
            # Filter by file_path if provided
            original_file_path = block.metadata.get("original_file_path")
            if source_regex is not None and (original_file_path is None or not re.search(source_regex, original_file_path,
                                                          0 if self.case_sensitive else re.IGNORECASE)):
                continue
            
            content = block.content
            if is_path_search:
                if block.file_path not in file_consts:
                    try:
                        with open(block.file_path, "r", encoding="utf-8") as f:
                            script_content = f.read()
                    except FileNotFoundError:
                        print(f"File not found: {block.file_path}")
                        continue
                    file_consts[block.file_path] = collect_constants(script_content)
                content = replace_constants_in_script(block.content, constants=file_consts[block.file_path])
                
            if regex.search(content):
                matches.append(
                    RetrievalResult(
                        chunk=CodeChunk(
                            content=content,
                            chunk_type="grep_match",
                            original_blocks=[block],
                            context="Grep match inside of GrepTool",
                            size_tokens=len(content) // self.config.get('chunker.chars_per_token', 4),
                            metadata={
                                "file_path": getattr(block, "file_path", None),
                                "original_file_path": block.metadata.get("original_file_path")
                            }
                        ),
                        score=1.0,            # constant score (grep has no similarity metric)
                        rank=1,
                        metadata={"pattern": pattern, "original_file_path": block.metadata.get("original_file_path")}
                    )
                )
        
        return matches

    def shorten_results(self, pattern: str, retrieval_results: List[RetrievalResult], limit: int) -> List[Dict[str, Any]]:
        """
        Shortens results by dynamically adjusting the context window (k) 
        so that the total number of lines fits within the 'limit'.

        Args:
            pattern (str): The regex pattern to highlight.
            retrieval_results (List[RetrievalResult]): The list of RetrievalResult objects to shorten.
            limit (int): The maximum allowed total number of lines across all results.

        Returns:
            List[Dict[str, Any]]: Each dict contains:
                - 'content': The shortened/formatted code string
                - 'line_start': Starting line number in the original file (1-indexed)
                - 'line_end': Ending line number in the original file (1-indexed)
        """
        
        clean_target = pattern.strip().strip("'\"").strip("/")
        
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(clean_target, flags)
        except re.error:
            regex = re.compile(re.escape(clean_target), flags)

        # 1. Pre-process: Identify all matching line numbers for every result
        # Store metadata needed to reconstruct line ranges
        file_data = []
        for result in retrieval_results:
            content = result.chunk.content
            lines = content.splitlines()
            matches = [i for i, line in enumerate(lines) if regex.search(line)]
            
            # If pattern not found (e.g. filename match only), treat line 0 as the 'match'
            if not matches and lines:
                matches = [0]
            
            if lines:
                # Get the original line range from the chunk
                orig_line_start, orig_line_end = result.chunk.get_line_range()
                file_data.append({
                    "lines": lines,
                    "matches": matches,
                    "orig_line_start": orig_line_start,
                    "orig_line_end": orig_line_end,
                })

        # 2. Helper: Calculate total lines used for a specific context size k
        def calculate_total_lines(k):
            total_lines = 0
            for item in file_data:
                lines_count = len(item["lines"])
                matches = item["matches"]
                
                # Determine the set of line indices to keep
                indices_to_keep = set()
                for m in matches:
                    start = max(0, m - k)
                    end = min(lines_count, m + k + 1)
                    indices_to_keep.update(range(start, end))
                
                if not indices_to_keep:
                    continue

                sorted_idx = sorted(indices_to_keep)
                
                # Count the lines we keep
                total_lines += len(sorted_idx)
                
                # Count the separator lines "..." (if gaps exist)
                prev = sorted_idx[0]
                for idx in sorted_idx[1:]:
                    if idx > prev + 1:
                        total_lines += 1 
                    prev = idx
                    
            return total_lines

        # 3. Optimize k: Find the largest k where total lines <= limit
        optimal_k = 0
        
        # First check if even 0 context fits
        if calculate_total_lines(0) <= limit:
            # Try increasing k until we hit the limit
            # We cap at 200 to prevent infinite loops (unlikely to need >200 lines context)
            prev_total = -1
            for k in range(1, 201):
                total = calculate_total_lines(k)
                
                if total > limit:
                    break # Previous k was the best
                
                if total == prev_total:
                    # Optimization: Increasing k didn't add lines (files are fully fully visible)
                    optimal_k = k
                    break
                
                optimal_k = k
                prev_total = total
        else:
            # Even k=0 is too big. We must stick to k=0 (pure matches).
            optimal_k = 0

        # 4. Render the final output using optimal_k, tracking displayed line ranges
        final_results = []
        for item in file_data:
            lines = item["lines"]
            matches = item["matches"]
            orig_line_start = item["orig_line_start"]
            
            indices = set()
            for m in matches:
                start = max(0, m - optimal_k)
                end = min(len(lines), m + optimal_k + 1)
                indices.update(range(start, end))
            
            sorted_idx = sorted(indices)
            if not sorted_idx:
                continue

            # Build the string with separators
            chunk_lines = []
            prev = sorted_idx[0]
            
            if prev > 0:
                chunk_lines.append("[truncated] ...")
            
            chunk_lines.append(lines[prev])
            
            for idx in sorted_idx[1:]:
                if idx > prev + 1:
                    chunk_lines.append("... [skipped] ...")
                chunk_lines.append(lines[idx])
                prev = idx
            
            if sorted_idx[-1] < len(lines)-1:
                chunk_lines.append("... [truncated]")
            
            # Calculate the displayed line range (1-indexed to match editor conventions)
            # First displayed line index (0-indexed) maps to orig_line_start + that index
            first_displayed_idx = sorted_idx[0]
            last_displayed_idx = sorted_idx[-1]
            
            # Filter out separator lines when calculating span
            display_line_start = orig_line_start + first_displayed_idx
            display_line_end = orig_line_start + last_displayed_idx
            
            final_results.append({
                "content": "\n".join(chunk_lines),
                "line_start": display_line_start,
                "line_end": display_line_end,
            })

        return final_results
    
    def _get_all_files(self, root_dirs: List[str]) -> List[str]:
        """
        Recursively gets the full path of all files in a directory and its subdirectories.

        Args:
            root_dir (str): The starting directory path.

        Returns:
            list: A list of absolute file paths.
        """
        all_files : set = set()
        for root_dir in root_dirs:
            # os.walk is the most efficient way to traverse a directory tree
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # 'dirpath' is the current directory we are in
                # 'filenames' is a list of all files in the current 'dirpath'

                for filename in filenames:
                    # os.path.join creates a full, correct path string
                    full_path = os.path.join(dirpath, filename)
                    all_files.add(full_path)

        return list(all_files)

    def build_save_index(self) -> None:
        """
        Build and save blocks.
        """
        self._parser = EnvisionParser()
        all_files = self._get_all_files(self.search_dirs)
        for file_path in all_files:
            original_path = self.mapping.get(os.path.splitext(os.path.basename(file_path))[0], None)
            blocks = self._parser.parse_file(file_path)
            for block in blocks:
                block.metadata["original_file_path"] = original_path
            self._blocks.extend(blocks)
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self._blocks, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save grep index: {e}")

    def load_index(self) -> None:
        """
        Load blocks from disk.
        """
        try:
            with open(self.index_path, "rb") as f:
                self._blocks = pickle.load(f)
            self._is_initialized = True
        except FileNotFoundError:
            raise FileNotFoundError(f"Grep index not found at {self.index_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load grep index: {e}")
    
    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="grep_tool",
            description=(
                "Exact-match text search across the codebase. "
                "Best for precise, unambiguous identifiers: a known function name, "
                "a variable, a file path, a specific string literal, or a unique keyword "
                "that is unlikely to appear in unrelated contexts. "
                "Avoid for broad concepts (e.g. 'forecasting', 'cost', 'order') — "
                "these will produce too many irrelevant matches. Use rag_tool instead."
            ),
            properties={
                "pattern": {
                    "type": "string",
                    "description": (
                        "A precise regex pattern to search for (most of the time a simple string suffices) in the CONTENT of the file. "
                        "E.g. 'LotMultiplier' or 'show (linechart|label)'."
                    ),
                },
                "sources": {
                    "type": "string",
                    "description": (
                        "Optional regex to restrict the search to files whose path matches. "
                        "Useful to search into some folders specifically (to limit the number of results for example). "
                        "This must be a regex, not a comma-separated list. "
                        "Use ONLY when necessary as you may miss relevant information otherwise. "
                        "E.g. '/Modules/' to search only inside the Modules folder."
                    ),
                },
                "block_type": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of block types to restrict the search to. "
                        "Allows targeting specific code structures -> especially useful to know when "
                        "resources are read or written."
                        "Valid values: COMMENT, SECTION_HEADER, IMPORT, READ, WRITE, CONST, EXPORT, "
                        "TABLE_DEFINITION, ASSIGNMENT, SHOW, KEEP_WHERE, FORM_READ, CONTROL_FLOW. "
                        "E.g. ['READ', 'FORM_READ'] to search only inside read statements."
                    ),
                },
            },
            required=["pattern"],
        )


if __name__ == "__main__":
    grep_tool = GrepTool()
    results = grep_tool.search(pattern='annual.*growth')
    print(len(results))
    for res in results:
        print(f"File: {res.metadata['original_file_path']}\nContent:\n{res.chunk.content}\n{'-'*40}\n")