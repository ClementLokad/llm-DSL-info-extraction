"""Syntactic search using grep-like operations"""
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class GrepResult:
    """Result from grep search"""
    files: List[str]
    count: int
    matches: List[Dict[str, str]]


class GrepSearcher:
    """Execute grep-based searches on source files"""
    
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs
        
    def search(self, pattern: str, case_sensitive: bool = False) -> GrepResult:
        """Search for pattern in source files"""
        matches = []
        files = set()
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
                                files.add(str(file_path))
                                matches.append({
                                    "file": str(file_path),
                                    "line": str(num),
                                    "content": line.strip()
                                })
                except Exception:
                    continue
                    
        return GrepResult(
            files=sorted(files),
            count=len(files),
            matches=matches[:100]
        )
    
    def format_answer(self, result: GrepResult, question: str) -> str:
        """Format grep result as answer"""
        if result.count == 0:
            return "Aucun résultat trouvé."
            
        if "combien" in question.lower():
            return str(result.count)
            
        return "\n".join(result.files)
