"""
Query session management for DSL Query System.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class QuerySession:
    """Represents a single query session with comprehensive tracking."""
    
    def __init__(self, query: str):
        """Initialize a new query session."""
        self.query = query
        self.timestamp = datetime.now()
        self.session_id = f"session_{int(time.time())}"
        
        # Processing data
        self.retrieved_chunks: List[Dict] = []
        self.selected_chunks: List[Dict] = []
        self.context_added: str = ""
        self.llm_input: str = ""
        self.llm_response: str = ""
        
        # Timing information
        self.timing: Dict[str, float] = {}
        
        # Processing steps
        self.steps: List[Dict[str, Any]] = []
        
    def add_step(self, step_name: str, data: Dict[str, Any], duration: float = 0.0) -> None:
        """Add a processing step to the session."""
        step = {
            'name': step_name,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'data': data
        }
        self.steps.append(step)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'query': self.query,
            'timestamp': self.timestamp.isoformat(),
            'retrieved_chunks': self.retrieved_chunks,
            'selected_chunks': self.selected_chunks,
            'context_added': self.context_added,
            'llm_input': self.llm_input,
            'llm_response': self.llm_response,
            'timing': self.timing,
            'steps': self.steps
        }
        
    def save_to_file(self, filepath: str) -> None:
        """Save session to JSON file."""
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'QuerySession':
        """Load session from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        session = cls(data['query'])
        session.session_id = data.get('session_id', session.session_id)
        session.timestamp = datetime.fromisoformat(data['timestamp'])
        session.retrieved_chunks = data.get('retrieved_chunks', [])
        session.selected_chunks = data.get('selected_chunks', [])
        session.context_added = data.get('context_added', '')
        session.llm_input = data.get('llm_input', '')
        session.llm_response = data.get('llm_response', '')
        session.timing = data.get('timing', {})
        session.steps = data.get('steps', [])
        
        return session
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        return {
            'session_id': self.session_id,
            'query': self.query,
            'timestamp': self.timestamp.isoformat(),
            'chunks_retrieved': len(self.retrieved_chunks),
            'chunks_selected': len(self.selected_chunks),
            'context_length': len(self.context_added),
            'response_length': len(self.llm_response),
            'total_duration': self.timing.get('total', 0.0),
            'steps_count': len(self.steps)
        }