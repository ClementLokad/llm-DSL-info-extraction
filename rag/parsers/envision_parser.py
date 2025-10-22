"""
Envision DSL parser for LOKAD's supply chain codebase.

This parser understands the structure and semantics of Envision DSL files (.nvn)
and extracts meaningful code blocks for embedding and retrieval.
"""

import re
from typing import List, Dict, Any, Optional
import logging

from rag.core.base_parser import BaseParser, CodeBlock

logger = logging.getLogger(__name__)

class EnvisionParser(BaseParser):
    """
    Parser specifically designed for Envision DSL (.nvn) files.
    
    The parser recognizes:
    - Comment blocks (/// style)
    - Read statements (data ingestion)
    - Table definitions and operations
    - Show statements (visualizations)
    - Variable assignments and calculations
    - Control structures (if/then/else, when, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from config_manager import get_config
        
        if config is None:
            # Load from global configuration
            global_config = get_config()
            self.config = global_config.get_parser_config()
        else:
            # Use provided configuration
            self.config = config
        
        # Initialize with base class if config was provided directly
        if config is not None:
            super().__init__(config)
        
        # Get regex compilation settings from config
        case_sensitive = self.config.get('case_sensitive', False)
        multiline_patterns = self.config.get('multiline_patterns', True)
        
        # Get section delimiter settings from config
        section_config = self.config.get('section_delimiter', {})
        min_chars = section_config.get('min_chars', 20)
        valid_chars_pattern = ''.join(section_config.get('valid_chars', ['~', '=', '-']))
        pattern_prefix = section_config.get('pattern_prefix', '///')
        
        # Compile regex patterns for better performance with config-driven flags
        flags = re.MULTILINE
        if multiline_patterns:
            flags |= re.DOTALL
        if not case_sensitive:
            flags |= re.IGNORECASE
        
        # Apply case sensitivity setting to all patterns
        comment_flags = re.MULTILINE
        if not case_sensitive:
            comment_flags |= re.IGNORECASE
        
        self._comment_block_pattern = re.compile(r'^///.*$', comment_flags)
        self._read_statement_pattern = re.compile(r'^read\s+.*?(?=^(?:read|show|table|\w+\s*=|///|$))', flags)
        self._table_definition_pattern = re.compile(r'^table\s+(\w+).*?(?=^(?:read|show|table|\w+\s*=|///|$))', flags)
        self._show_statement_pattern = re.compile(r'^show\s+.*?(?=^(?:read|show|table|\w+\s*=|///|$))', flags)
        self._assignment_pattern = re.compile(r'^(\w+(?:\.\w+)*)\s*=.*?(?=^(?:read|show|table|\w+\s*=|///|$))', flags)
        
        # Dynamic section delimiter pattern from config
        delimiter_pattern = f'^{re.escape(pattern_prefix)}[{re.escape(valid_chars_pattern)}]{{{min_chars},}}.*$'
        delimiter_flags = re.MULTILINE
        if not case_sensitive:
            delimiter_flags |= re.IGNORECASE
        self._section_delimiter_pattern = re.compile(delimiter_pattern, delimiter_flags)
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return supported file extensions from configuration."""
        return self.config.get('supported_extensions', ['.nvn'])
    
    def parse_file(self, file_path: str) -> List[CodeBlock]:
        """Parse an Envision file and extract code blocks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_content(content, file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Envision file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing Envision file {file_path}: {e}")
    
    def parse_content(self, content: str, file_path: str = "") -> List[CodeBlock]:
        """Parse Envision content and extract code blocks."""
        blocks = []
        lines = content.split('\n')
        
        # First, identify major sections using delimiter comments
        sections = self._identify_sections(content, lines)
        
        # Parse each section
        for section_name, section_content, start_line, end_line in sections:
            section_blocks = self._parse_section(section_content, section_name, start_line, file_path)
            
            # Validate that blocks don't exceed the section boundaries
            for block in section_blocks:
                if block.line_end > end_line:
                    logger.warning(f"Block {block.name or 'unnamed'} extends beyond section {section_name} boundary (line {block.line_end} > {end_line})")
                    # Adjust the block's end line to not exceed section boundary
                    block.line_end = min(block.line_end, end_line)
            
            blocks.extend(section_blocks)
        
        # If no major sections found, parse the entire content as one section
        if not sections:
            blocks = self._parse_section(content, "main", 0, file_path)
        
        return blocks
    
    def _identify_sections(self, content: str, lines: List[str]) -> List[tuple]:
        """Identify major sections in the Envision file."""
        sections = []
        current_section_start = 0
        current_section_name = "header"
        
        # Use finditer on content for more consistent regex matching
        delimiter_matches = list(self._section_delimiter_pattern.finditer(content))
        
        if not delimiter_matches:
            # No section delimiters found, return empty to parse as single section
            return []
        
        for match in delimiter_matches:
            # Find which line this match is on
            match_line_num = content[:match.start()].count('\n')
            delimiter_line = lines[match_line_num]
            
            # Save previous section if it has content
            if match_line_num > current_section_start:
                section_content = '\n'.join(lines[current_section_start:match_line_num])
                if section_content.strip():
                    sections.append((
                        current_section_name,
                        section_content,
                        current_section_start,
                        match_line_num - 1
                    ))
            
            # Extract section name from the delimiter comment
            section_name = self._extract_section_name(delimiter_line)
            current_section_name = section_name
            current_section_start = match_line_num + 1
        
        # Add the last section
        if current_section_start < len(lines):
            section_content = '\n'.join(lines[current_section_start:])
            if section_content.strip():
                sections.append((
                    current_section_name,
                    section_content,
                    current_section_start,
                    len(lines) - 1
                ))
        
        return sections
    
    def _extract_section_name(self, delimiter_line: str) -> str:
        """Extract section name from a delimiter comment line."""
        # Remove /// and delimiter characters, extract the text in the middle
        cleaned = re.sub(r'^///[~=-]*', '', delimiter_line)
        cleaned = re.sub(r'[~=-]*$', '', cleaned)
        section_name = cleaned.strip()
        
        if not section_name:
            return "section"
        
        # We remove non alphanumeric caracters
        section_name = re.sub(r'[^a-zA-Z0-9]+', '_', section_name).strip('_').lower()
        return section_name or "section"
    
    def _parse_section(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse a single section and extract code blocks."""
        blocks = []
        
        # Parse different types of blocks in order of specificity
        blocks.extend(self._parse_read_statements(content, section_name, base_line, file_path))
        blocks.extend(self._parse_table_definitions(content, section_name, base_line, file_path))
        blocks.extend(self._parse_show_statements(content, section_name, base_line, file_path))
        blocks.extend(self._parse_assignments(content, section_name, base_line, file_path))
        blocks.extend(self._parse_comment_blocks(content, section_name, base_line, file_path))
        
        return blocks
    
    def _parse_read_statements(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse read statements (data ingestion)."""
        blocks = []
        
        for match in self._read_statement_pattern.finditer(content):
            read_content = match.group(0).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + base_line
            end_line = content[:end_pos].count('\n') + base_line
            
            # Extract table name
            table_name = self._extract_read_table_name(read_content)
            
            block = CodeBlock(
                content=read_content,
                block_type="read_statement",
                name=table_name,
                line_start=start_line,
                line_end=end_line,
                file_path=file_path,
                metadata={
                    "section": section_name,
                    "table_name": table_name,
                    "is_data_ingestion": True
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _parse_table_definitions(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse table definitions and operations."""
        blocks = []
        
        for match in self._table_definition_pattern.finditer(content):
            table_content = match.group(0).strip()
            table_name = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + base_line
            end_line = content[:end_pos].count('\n') + base_line
            
            block = CodeBlock(
                content=table_content,
                block_type="table_definition",
                name=table_name,
                line_start=start_line,
                line_end=end_line,
                file_path=file_path,
                metadata={
                    "section": section_name,
                    "table_name": table_name,
                    "is_table_operation": True
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _parse_show_statements(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse show statements (visualizations and output)."""
        blocks = []
        
        for match in self._show_statement_pattern.finditer(content):
            show_content = match.group(0).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + base_line
            end_line = content[:end_pos].count('\n') + base_line
            
            # Extract show type and name
            show_type, show_name = self._extract_show_info(show_content)
            
            block = CodeBlock(
                content=show_content,
                block_type="show_statement",
                name=show_name,
                line_start=start_line,
                line_end=end_line,
                file_path=file_path,
                metadata={
                    "section": section_name,
                    "show_type": show_type,
                    "show_name": show_name,
                    "is_visualization": True
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _parse_assignments(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse variable assignments and calculations."""
        blocks = []
        
        for match in self._assignment_pattern.finditer(content):
            assignment_content = match.group(0).strip()
            variable_name = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            
            # Skip if this is part of a read, table, or show statement
            if any(keyword in assignment_content.lower()[:20] for keyword in ['read ', 'table ', 'show ']):
                continue
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + base_line
            end_line = content[:end_pos].count('\n') + base_line
            
            # Determine assignment type
            assignment_type = self._classify_assignment(assignment_content)
            
            block = CodeBlock(
                content=assignment_content,
                block_type="assignment",
                name=variable_name,
                line_start=start_line,
                line_end=end_line,
                file_path=file_path,
                metadata={
                    "section": section_name,
                    "variable_name": variable_name,
                    "assignment_type": assignment_type,
                    "is_calculation": True
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _parse_comment_blocks(self, content: str, section_name: str, base_line: int, file_path: str) -> List[CodeBlock]:
        """Parse comment blocks for documentation."""
        blocks = []
        lines = content.split('\n')
        
        current_comment = []
        comment_start_line = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('///') and not self._section_delimiter_pattern.match(line.strip()):
                if not current_comment:
                    comment_start_line = i
                current_comment.append(line.strip())
            else:
                if current_comment:
                    # End of comment block
                    comment_content = '\n'.join(current_comment)
                    
                    block = CodeBlock(
                        content=comment_content,
                        block_type="comment_block",
                        name=None,
                        line_start=base_line + comment_start_line,
                        line_end=base_line + i - 1,
                        file_path=file_path,
                        metadata={
                            "section": section_name,
                            "is_documentation": True,
                            "comment_type": "block"
                        }
                    )
                    blocks.append(block)
                    current_comment = []
        
        # Handle comment at end of file
        if current_comment:
            comment_content = '\n'.join(current_comment)
            block = CodeBlock(
                content=comment_content,
                block_type="comment_block",
                name=None,
                line_start=base_line + comment_start_line,
                line_end=base_line + len(lines) - 1,
                file_path=file_path,
                metadata={
                    "section": section_name,
                    "is_documentation": True,
                    "comment_type": "block"
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _extract_read_table_name(self, read_content: str) -> Optional[str]:
        """Extract table name from a read statement."""
        # Look for pattern: read "path" as TableName
        flags = 0 if self.config.get('case_sensitive', False) else re.IGNORECASE
        match = re.search(r'as\s+(\w+)', read_content, flags)
        if match:
            return match.group(1)
        return None
    
    def _extract_show_info(self, show_content: str) -> tuple:
        """Extract show type and name from a show statement."""
        # Look for patterns like: show table "Name", show label "Name"
        flags = 0 if self.config.get('case_sensitive', False) else re.IGNORECASE
        type_match = re.search(r'show\s+(\w+)', show_content, flags)
        show_type = type_match.group(1) if type_match else "unknown"
        
        name_match = re.search(r'show\s+\w+\s+"([^"]+)"', show_content, flags)
        show_name = name_match.group(1) if name_match else None
        
        return show_type, show_name
    
    def _classify_assignment(self, assignment_content: str) -> str:
        """Classify the type of assignment based on its content."""
        # Use config to determine if case-sensitive matching should be used
        case_sensitive = self.config.get('case_sensitive', False)
        content = assignment_content if case_sensitive else assignment_content.lower()
        
        # Define patterns based on case sensitivity
        aggregation_funcs = ['same(', 'max(', 'min(', 'sum(', 'avg(', 'mode(']
        conditional_keywords = ['if ', 'then ', 'else ', 'when(']
        calculation_keywords = ['+', '-', '*', '/', 'random']
        date_keywords = ['date(', 'today']
        
        if not case_sensitive:
            aggregation_funcs = [func.lower() for func in aggregation_funcs]
            conditional_keywords = [kw.lower() for kw in conditional_keywords]
            date_keywords = [kw.lower() for kw in date_keywords]
        
        if any(func in content for func in aggregation_funcs):
            return "aggregation"
        elif any(keyword in content for keyword in conditional_keywords):
            return "conditional"
        elif any(keyword in content for keyword in calculation_keywords):
            return "calculation"
        elif ('concat(' if case_sensitive else 'concat(') in content:
            return "string_operation"
        elif any(keyword in content for keyword in date_keywords):
            return "date_operation"
        else:
            return "simple_assignment"
    
    def extract_dependencies(self, code_block: CodeBlock) -> List[str]:
        """Extract dependencies (table/variable references) from a code block."""
        dependencies = []
        content = code_block.content
        
        # Extract table references (e.g., Catalog.ItemCode)
        table_refs = re.findall(r'(\w+)\.', content)
        dependencies.extend(list(set(table_refs)))
        
        # Extract function calls
        function_refs = re.findall(r'(\w+)\(', content)
        dependencies.extend(list(set(function_refs)))
        
        return list(set(dependencies))
