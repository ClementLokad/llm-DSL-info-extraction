"""Syntactic search using grep-like operations"""
import re
import os
from pathlib import Path
from typing import List
from rag.core.base_retriever import RetrievalResult
from rag.core.base_chunker import CodeChunk
from utils.get_mapping import get_file_mapping
from rag.utils.script_scanner import scan_script_for_references


class GrepRetriever:
    """Execute grep-based searches on source files"""
    
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs
        self.mapping = get_file_mapping()
        
    def search(self, pattern: str, case_sensitive: bool = False) -> List[RetrievalResult]:
        """Search for pattern in source files"""
        matches = []
        
        # Check if pattern looks like a file path or we are searching for specific IO operations
        # Heuristic: if pattern looks like a path (slashes, extensions, etc.)
        # Covers cases like: /1. utilities/2. preprocess/[ 3 ] - Forecast Autodiff, /Clean/Items.ion
        path_regex = r"(\/|\\)|(\.[a-zA-Z0-9]+$)"
        is_path_search = bool(re.search(path_regex, pattern))
        
        if is_path_search:
             return self._smart_path_search(pattern)
        
        # Fallback to standard regex grep
        return self._standard_grep(pattern, case_sensitive)

    def _smart_path_search(self, target_path: str) -> List[RetrievalResult]:
        """Use advanced scanner to resolve constants and find file references"""
        matches = []
        # Remove regex escaping if present (since router might escape it)
        clean_target = target_path.replace(r"\.", ".").replace("\\", "")
        
        for dir_path in self.search_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
                
            for file_path in path.rglob("*.nvn"):
                hits = scan_script_for_references(file_path, clean_target)
                if not hits:
                    continue
                
                original_path = self._get_original_path(file_path)
                
                for hit in hits:
                     matches.append(RetrievalResult(
                        chunk=CodeChunk(
                            content=hit['raw'],
                            chunk_type="smart_reference",
                            metadata={
                                "original_file_path": original_path,
                                "line_number": hit['line'],
                                "verb": hit['verb'],
                                "resolved_path": hit['resolved_path']
                            }
                        ),
                        score=1.0, # High confidence for exact resolved matches
                        rank=1
                    ))
        return matches

    def _standard_grep(self, pattern: str, case_sensitive: bool) -> List[RetrievalResult]:
        """Standard line-by-line regex search"""
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for dir_path in self.search_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
                
            for file_path in path.rglob("*.nvn"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        original_path = None
                        # Check first line for mapping/metadata if needed, 
                        # though we can reuse _get_original_path logic if we want consistency.
                        # For speed in standard grep, we might just read linear.
                        
                        lines = f.readlines()
                        
                        for num, line in enumerate(lines):
                            if re.search(pattern, line, flags):
                                if original_path is None:
                                     # try lazy load
                                     original_path = self._get_original_path(file_path)

                                matches.append(RetrievalResult(
                                    chunk=CodeChunk(content=line.strip(),
                                                    chunk_type="grep_match",
                                                    metadata={"original_file_path": original_path,
                                                              "line_number": num}),
                                    score=None,
                                    rank=None
                                ))
                except Exception:
                    continue
        return matches

    def _get_original_path(self, file_path: Path) -> str:
        """Helper to retrieve original path from mapping or file header"""
        # 1. Try mapping from filename stem
        stem = os.path.splitext(os.path.basename(file_path))[0]
        mapped = self.mapping.get(stem)
        if mapped:
            return mapped
            
        # 2. Try reading header
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                if first_line.startswith("///ORIGINAL_PATH: "):
                    return first_line.replace("///ORIGINAL_PATH: ", "").strip()
        except:
            pass
            
        return str(file_path)
    
    # def format_answer(self, result: GrepResult, question: str) -> str:
    #     """Format grep result as answer"""
    #     if result.count == 0:
    #         return "Aucun résultat trouvé."
            
    #     if "combien" in question.lower():
    #         return str(result.count)
            
    #     return "\n".join(result.files)

