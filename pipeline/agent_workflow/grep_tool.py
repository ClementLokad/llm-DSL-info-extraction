from importlib import metadata
import re
import pickle
import os
from typing import List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.core.base_retriever import RetrievalResult, CodeChunk
from rag.core.base_parser import CodeBlock
from rag.parsers.envision_parser import EnvisionParser
from pipeline.agent_workflow.workflow_base import BaseGrepTool
from get_mapping import get_file_mapping
from config_manager import get_config


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
        sources: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Here query_embedding is misused: it must contain the search pattern as a string.
        This allows full compatibility with BaseRetriever.

        Convention:
            - query_embedding is a numpy array of shape (1,) containing a string
        """
        
        if sources is not None:
            valid = False
            for s in sources:
                if self.check_suffix_match(s.strip(), list(self.mapping.values()), inverse=True):
                    valid = True
                    break
            if not valid:
                sources = None  # Ignore invalid source filters
            

        pattern = pattern.strip().strip("/")

        regex = re.compile(pattern, 0 if self.case_sensitive else re.IGNORECASE)

        results = []
        rank = 1

        for block in self._blocks:
            # Filter by file_path if provided
            if sources is not None and not self.check_suffix_match(block.metadata["original_file_path"], sources):
                continue

            # Match on the full chunk content
            if regex.search(block.content):
                results.append(
                    RetrievalResult(
                        chunk=CodeChunk(
                            content=block.content,
                            chunk_type=block.block_type,
                            original_blocks=[block],
                            context="Grep match",
                            size_tokens=len(block.content) // self.config.get('chunker.chars_per_token', 4),
                            metadata={"file_path": getattr(block, "file_path", None),
                                      "original_file_path": block.metadata["original_file_path"]}
                        ),
                        score=1.0,            # constant score (grep has no similarity metric)
                        rank=rank,
                        metadata={"pattern": pattern, "original_file_path": block.metadata["original_file_path"]}
                    )
                )
                rank += 1

        return results
    
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

if __name__ == "__main__":
    grep_tool = GrepTool()
    results = grep_tool.search(pattern="Clean/Items.ion", sources=["1 - Item Inspector"])
    for res in results:
        print(f"File: {res.metadata['original_file_path']}\n")#Content:\n{res.chunk.content}\n{'-'*40}\n")