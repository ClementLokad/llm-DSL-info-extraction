"""Syntactic search using grep-like operations"""
import re
from pathlib import Path
from typing import List
from rag.core.base_retriever import RetrievalResult
from rag.core.base_chunker import CodeChunk


class GrepRetriever:
    """Execute grep-based searches on source files"""
    
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs
        
    def search(self, pattern: str, case_sensitive: bool = False) -> List[RetrievalResult]:
        """Search for pattern in source files"""
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for dir_path in self.search_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
                
            for file_path in path.rglob("*.nvn"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for num, line in enumerate(f, 1):
                            if re.search(pattern, line, flags):
                                matches.append(RetrievalResult(
                                    chunk=CodeChunk(content=line.strip(),
                                                    chunk_type="grep_match"),
                                    metadata={
                                        "file_path": str(file_path),
                                        "line_number": num
                                    },
                                    score=None,
                                    rank=None
                                ))
                except Exception:
                    continue
                    
        return matches
    
    # def format_answer(self, result: GrepResult, question: str) -> str:
    #     """Format grep result as answer"""
    #     if result.count == 0:
    #         return "Aucun résultat trouvé."
            
    #     if "combien" in question.lower():
    #         return str(result.count)
            
    #     return "\n".join(result.files)

