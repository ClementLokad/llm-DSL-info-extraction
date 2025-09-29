"""
Module for intelligent chunking of DSL scripts.
"""
from typing import List, Dict, Union
from dataclasses import dataclass
import re

@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_line: int
    end_line: int
    metadata: Dict[str, Union[str, int]]

class DSLChunker:
    """Handles intelligent chunking of DSL scripts based on semantic boundaries."""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Find semantic boundaries in the DSL code (sections, major blocks, etc.)."""
        # Add minimum chunk size to avoid single-line chunks
        min_chunk_size = 5
        boundaries = []
        
        # Look for section comments marked with ///~~~~ or ///----
        section_pattern = r'^///[-~]{3,}$|^///\s*[-]{3,}[A-Za-z\s]+[-]{3,}'
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.match(section_pattern, line.strip()):
                if not boundaries or (i - boundaries[-1]) >= min_chunk_size:
                    boundaries.append(i)
        
        # Ensure we start from the beginning of the file
        if not boundaries or boundaries[0] > min_chunk_size:
            boundaries.insert(0, 0)
            
        # Ensure we capture the end of the file
        if boundaries[-1] < len(lines) - min_chunk_size:
            boundaries.append(len(lines))
                
        return boundaries

    def _create_chunk_metadata(self, text: str, start_line: int, end_line: int) -> Dict[str, Union[str, int]]:
        """Create metadata for a chunk including important identifiers and context."""
        metadata = {
            'start_line': start_line,
            'end_line': end_line,
            'length': end_line - start_line + 1,
        }
        
        # Extract table names and key identifiers
        table_pattern = r'table\s+([A-Za-z][A-Za-z0-9_]*)'
        tables = re.findall(table_pattern, text)
        if tables:
            metadata['tables'] = tables
            
        # Extract read statements and input files
        read_pattern = r'read\s+"([^"]+)"'
        reads = re.findall(read_pattern, text)
        if reads:
            metadata['input_files'] = reads
        
        # Add DSL-specific elements
        metadata['dsl_elements'] = []
        
        if 'show' in text:
            metadata['dsl_elements'].append('visualization')
        if 'where' in text:
            metadata['dsl_elements'].append('filter')
        if re.search(r'match|when|if', text):
            metadata['dsl_elements'].append('conditional')
        if re.search(r'group\s+by', text, re.IGNORECASE):
            metadata['dsl_elements'].append('aggregation')
        if re.search(r'order\s+by', text, re.IGNORECASE):
            metadata['dsl_elements'].append('sorting')
        
        # Extract constants and variables
        const_pattern = r'const\s+([A-Za-z][A-Za-z0-9_]*)\s*='
        constants = re.findall(const_pattern, text)
        if constants:
            metadata['constants'] = constants
            
        # Count DSL operations
        metadata['operation_counts'] = {
            'show': len(re.findall(r'\bshow\b', text)),
            'where': len(re.findall(r'\bwhere\b', text)),
            'keep': len(re.findall(r'\bkeep\b', text)),
            'read': len(re.findall(r'\bread\b', text))
        }
            
        return metadata

    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into semantic chunks based on DSL structure."""
        boundaries = self._find_semantic_boundaries(text)
        chunks = []
        lines = text.split('\n')
        
        if not boundaries:
            # If no semantic boundaries found, use size-based chunking
            chunk_size = min(self.max_chunk_size, len(lines))
            for i in range(0, len(lines), chunk_size):
                end_idx = min(i + chunk_size, len(lines))
                chunk_text = '\n'.join(lines[i:end_idx])
                metadata = self._create_chunk_metadata(chunk_text, i, end_idx-1)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_line=i,
                    end_line=end_idx-1,
                    metadata=metadata
                ))
        else:
            # Use semantic boundaries for chunking
            for i in range(len(boundaries)):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
                chunk_text = '\n'.join(lines[start_idx:end_idx])
                metadata = self._create_chunk_metadata(chunk_text, start_idx, end_idx-1)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_line=start_idx,
                    end_line=end_idx-1,
                    metadata=metadata
                ))
                
        return chunks