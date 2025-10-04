"""
Base parser interface for code parsing and analysis.

This module defines the abstract interface that all code parsers must implement
to ensure consistent behavior across different views of Envision DSL.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeBlock:
    """
    Represents a parsed code block with metadata.
    
    Attributes:
        content: The raw code content
        block_type: Type of code block (function, table, expression, etc.)
        name: Identifier/name of the block if available
        line_start: Starting line number in source file
        line_end: Ending line number in source file
        file_path: Path to the source file
        dependencies: List of dependencies/imports this block uses
        metadata: Additional parser-specific metadata
    """
    content: str
    block_type: str
    name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    file_path: str = ""
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

class BaseParser(ABC):
    """
    Abstract base class for all code parsers.
    
    This class defines the interface that all parsers must implement to ensure
    consistent behavior across different languages and DSLs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser with configuration.
        
        Args:
            config: Parser-specific configuration options
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Return list of file extensions this parser supports.
        
        Returns:
            List of supported file extensions (e.g., ['.nvn', ...])
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: str) -> List[CodeBlock]:
        """
        Parse a single file and extract code blocks.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of parsed code blocks
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass
    
    @abstractmethod
    def parse_content(self, content: str, file_path: str = "") -> List[CodeBlock]:
        """
        Parse content string and extract code blocks.
        
        Args:
            content: Raw code content to parse
            file_path: Optional path for metadata (useful for imports/references)
            
        Returns:
            List of parsed code blocks
            
        Raises:
            ValueError: If content format is invalid
        """
        pass
    
    def validate_syntax(self, content: str) -> bool:
        """
        Validate if the content has valid syntax for this language.
        
        Args:
            content: Code content to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            self.parse_content(content)
            return True
        except Exception as e:
            self.logger.debug(f"Syntax validation failed: {e}")
            return False
    
    def extract_dependencies(self, code_block: CodeBlock) -> List[str]:
        """
        Extract dependencies from a code block.
        
        Args:
            code_block: Code block to analyze
            
        Returns:
            List of dependency identifiers
        """
        # Default implementation - can be overridden by specific parsers
        return code_block.dependencies or []
    
    def get_block_signature(self, code_block: CodeBlock) -> str:
        """
        Generate a signature/summary for a code block.
        
        Args:
            code_block: Code block to summarize
            
        Returns:
            String signature of the block
        """
        # Default implementation - can be overridden by specific parsers
        if code_block.name:
            return f"{code_block.block_type}: {code_block.name}"
        else:
            return f"{code_block.block_type} (lines {code_block.line_start}-{code_block.line_end})"