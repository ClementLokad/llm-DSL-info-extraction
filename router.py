"""Query router - classifies and routes queries to appropriate executor"""
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QueryType(Enum):
    """Query execution type"""
    GREP = "grep"
    RAG = "rag"


@dataclass  
class Classification:
    """Query classification result"""
    qtype: QueryType
    confidence: float
    pattern: Optional[str] = None  # For GREP


class Router:
    """Routes queries to GREP or RAG based on classification"""
    
    # Heuristic patterns
    GREP_KEYWORDS = [
        r"\bquels? scripts?\b", r"\bcombien\b", 
        r"\bliste\b", r"\boù figure\b", r"\bqui (lit|écrit|utilise)\b"
    ]
    
    def __init__(self, agent=None):
        self.agent = agent
        
    def classify(self, question: str) -> Classification:
        """Classify question as GREP or RAG"""
        # Quick heuristic classification
        q_lower = question.lower()
        
        # File path or variable explicitly mentioned → GREP
        if re.search(r"/[\w/\-\.]+\.\w+", question):
            pattern = self._extract_pattern(question)
            return Classification(QueryType.GREP, 0.95, pattern)
            
        # Strong GREP keywords
        if any(re.search(kw, q_lower) for kw in self.GREP_KEYWORDS):
            pattern = self._extract_pattern(question)
            return Classification(QueryType.GREP, 0.85, pattern)
            
        # Default to RAG for semantic questions
        return Classification(QueryType.RAG, 0.8)
        
    def _extract_pattern(self, question: str) -> str:
        """Extract search pattern from question"""
        # Extract file path
        file_match = re.search(r"/[\w/\-\.]+\.\w+", question)
        if file_match:
            return re.escape(file_match.group(0)).replace(r"\.", r"\.")
            
        # Extract CamelCase variable
        var_match = re.search(r"\b[A-Z][a-zA-Z0-9]+\b", question)
        if var_match:
            return var_match.group(0)
            
        # Extract quoted term
        quote_match = re.search(r'["\'](.+?)["\']', question)
        if quote_match:
            return quote_match.group(1)
            
        return ""
