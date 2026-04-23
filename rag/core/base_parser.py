"""
Base parser interface for code parsing and analysis.

This module defines the abstract interface that all code parsers must implement
to ensure consistent behavior across different views of Envision DSL.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from utils.config_manager import get_config
from rag.utils.handle_tokens import get_token_count
import logging

logger = logging.getLogger(__name__)

class BlockType(Enum):
    """Enumeration of different block types in Envision scripts."""
    COMMENT = "comment"
    SECTION_HEADER = "section_header"
    IMPORT = "import"
    READ = "read"
    WRITE = "write"
    CONST = "const"
    EXPORT = "export"
    TABLE_DEFINITION = "table_definition"
    ASSIGNMENT = "assignment"
    SHOW = "show"
    KEEP_WHERE = "keep_where"
    FORM_READ = "form_read"
    CONTROL_FLOW = "control_flow"
    UNKNOWN = "unknown"

@dataclass
class CodeBlock:
    """
    Represents a parsed code block with metadata.
    
    Attributes:
        content: The raw code content
        block_type: Type of code block (function, table, expression, etc.)
        name: Identifier/name of the block if available (ex: table name for a table definition)
        line_start: Starting line number in source file
        line_end: Ending line number in source file
        file_path: Path to the physical source file
        dependencies: List of dependencies/imports this block uses
        metadata: Additional parser-specific metadata
    """
    content: str
    block_type: BlockType
    name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    file_path: str = ""
    dependencies: Set[str] = None
    definitions: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.metadata is None:
            self.metadata = {}
        if self.definitions is None:
            self.definitions = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the CodeBlock to a dictionary for JSON serialization."""
        return {
            'content': self.content,
            'block_type': self.block_type,
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'file_path': self.file_path,
            'dependencies': self.dependencies,
            'definitions': self.definitions,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeBlock':
        """
        Reconstruct a CodeBlock instance from its dictionary representation.
        
        Args:
            data: Dictionary containing CodeBlock serialization (typically from to_dict())
            
        Returns:
            A new CodeBlock instance with the same data
            
        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If block_type value is not a valid BlockType
        """
        # Handle block_type: convert from enum or string to BlockType
        block_type_val = data.get('block_type')
        if isinstance(block_type_val, BlockType):
            block_type = block_type_val
        elif isinstance(block_type_val, str):
            # Try to get enum by name or by value
            try:
                block_type = BlockType[block_type_val]
            except KeyError:
                try:
                    block_type = BlockType(block_type_val)
                except ValueError:
                    raise ValueError(f"Invalid block_type: {block_type_val}")
        else:
            raise ValueError(f"block_type must be BlockType or string, got {type(block_type_val)}")
        
        # Convert dependencies and definitions from lists to sets if needed
        dependencies = data.get('dependencies')
        if dependencies is not None and not isinstance(dependencies, set):
            dependencies = set(dependencies) if dependencies else None
        
        definitions = data.get('definitions')
        if definitions is not None and not isinstance(definitions, set):
            definitions = set(definitions) if definitions else None
        
        return cls(
            content=data.get('content', ''),
            block_type=block_type,
            name=data.get('name'),
            line_start=data.get('line_start', 0),
            line_end=data.get('line_end', 0),
            file_path=data.get('file_path', ''),
            dependencies=dependencies,
            definitions=definitions,
            metadata=data.get('metadata', {}),
        )
    
    def __len__(self) -> int:
        """Returns number of lines in the block."""
        return self.content.count('\n') + 1
    
    def get_token_count(self, encoding_name: str = "cl100k_base") -> int:
        """Returns approximate token count for this block."""
        return get_token_count(self.content, encoding_name)

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
        self.config = config or get_config().get('parser', {})
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
