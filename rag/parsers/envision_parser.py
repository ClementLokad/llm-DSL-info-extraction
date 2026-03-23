import re
from typing import List, Set, Optional, Tuple, Dict, Any
from rag.core.base_parser import BlockType, CodeBlock, BaseParser
from config_manager import get_config


class EnvisionParser(BaseParser):
    """
    Parser for Envision scripts that identifies semantic blocks.
    """
    
    # Regex patterns for different block types
    PATTERNS = {
        'section_header': re.compile(r'^///\s*[~=\-]{3,}'),
        'comment': re.compile(r'^///|^//'),
        'import': re.compile(r'^\s*import\s+'),
        'read': re.compile(r'^\s*read\s+'),
        'write': re.compile(r'^\s*write\s+'),
        'const': re.compile(r'^\s*const\s+'),
        'export': re.compile(r'^\s*export\s+'),
        'table': re.compile(r'^\s*table\s+'),
        'show': re.compile(r'^\s*show\s+'),
        'keep': re.compile(r'^\s*keep\s+'),
        'where': re.compile(r'^\s*where\s+'),
        'form_read': re.compile(r'^\s*read\s+form\s+'),
    }
    
    # Pattern to extract variable/table names from assignments
    ASSIGNMENT_PATTERN = re.compile(r'^\s*(\w+(?:\.\w+)?)\s*=')
    
    # Envision built-in functions and keywords (not dependencies)
    BUILTINS = {
        # Aggregation functions
        'sum', 'max', 'min', 'avg', 'count', 'distinct', 'mode', 'median',
        # Date/time functions
        'date', 'year', 'month', 'week', 'day', 'monday', 'today',
        # Math functions
        'abs', 'round', 'floor', 'ceiling', 'sqrt', 'exp', 'log', 'pow',
        # String functions
        'concat', 'replace', 'substr', 'strlen', 'split',
        # Table functions
        'cross', 'extend', 'range', 'slice',
        # Statistical functions
        'random', 'uniform', 'poisson', 'normal',
        # Logical
        'if', 'then', 'else', 'when', 'where', 'not', 'and', 'or', 'all', 'any', 'assert',
        # Aggregation keywords
        'by', 'at', 'into', 'scan', 'same', 'default', 'group', 'order',
        # Special
        'as', 'with', 'expect', 'unsafe', 'keep', 'show', 'upload', 'form',
        # Functions
        'cumsum', 'rankd', 'rankrev', 'sliceSearchUrl', 'dashUrl', 'downloadUrl',
        # Types
        'text', 'number', 'date', 'boolean', 'ranvar', 'zedfunc',
        # Colors
        'rgb', 'rgba',
    }
    
    # Common Envision table properties (not separate dependencies)
    COMMON_PROPERTIES = {
        'N', 'Id', 'date', 'week', 'month', 'year', 'day',
    }
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return supported file extensions from configuration."""
        return self.config.get('supported_extensions', ['.nvn'])
    
    def parse_content(self, content: str, file_path: str = "") -> List[CodeBlock]:
        """
        Parse an Envision script into semantic blocks.
        
        Args:
            content: The full content of the Envision script
            
        Returns:
            List of CodeBlock objects
        """
        lines = content.split('\n')
        blocks = []
        i = 0
        
        while i < len(lines):
            block, lines_consumed = self._parse_block(lines, i)
            if block:
                block.file_path = file_path
                blocks.append(block)
            i += lines_consumed
        
        return blocks

    def parse_file(self, file_path: str) -> List[CodeBlock]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_content(content, file_path=file_path)
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def _extract_table_references(self, code: str) -> Set[str]:
        """
        Extract table references from code, being careful to avoid false positives.
        
        Returns only the base table names (e.g., "Items" from "Items.Field")
        Properly handles Table.Field references to extract only "Table"
        """
        refs = set()
        
        # Remove string literals first to avoid matching content inside strings
        code_no_strings = self._remove_all_strings(code)
        
        # First pass: Find all Table.field patterns and collect both tables and fields
        table_field_pattern = re.compile(r'\b([a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)')
        field_names = set()  # Track field names to exclude them later
        
        for match in table_field_pattern.finditer(code_no_strings):
            table_name = match.group(1)
            field_name = match.group(2)
            
            field_names.add(field_name)  # Remember this is a field, not a table
            
            if self._is_likely_table_name(table_name):
                refs.add(table_name)
        
        # Second pass: Look for standalone identifiers, but only if:
        # 1. They weren't seen as field names
        # 2. They appear in a context suggesting table usage
        standalone_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b(?!\s*\.)')
        
        for match in standalone_pattern.finditer(code_no_strings):
            identifier = match.group(1)
            
            # Skip if it's a field name from Table.field pattern
            if identifier in field_names:
                continue
            
            # Skip if it's a builtin
            if identifier.lower() in self.BUILTINS:
                continue
            
            # Skip if it's a common property
            if identifier in self.COMMON_PROPERTIES:
                continue
            
            # Get context to check if this is really a table reference
            start = max(0, match.start() - 30)
            end = min(len(code_no_strings), match.end() + 30)
            context = code_no_strings[start:end]
            
            # Skip if it appears in a type definition (: Type)
            if re.search(r':\s*' + re.escape(identifier) + r'\s*(?:$|\n|,)', context):
                continue
            
            # Skip if it's after 'as' (likely a table being defined, not referenced)
            if re.search(r'\bas\s+' + re.escape(identifier) + r'\b', context):
                continue
            
            # Skip if it's before '=' at start of line (it's being defined)
            line_start = code_no_strings.rfind('\n', 0, match.start())
            if line_start == -1:
                line_start = 0
            line_end = code_no_strings.find('\n', match.end(), len(code_no_strings))
            if line_end == -1:
                line_end = len(code_no_strings)
            line_portion = code_no_strings[line_start:line_end]
            if re.match(r'^\s*' + re.escape(identifier) + r'\s*=', line_portion.lstrip()):
                continue

            if re.match(r'.*//.*' + re.escape(identifier), line_portion.lstrip()):
                continue # Skip if in single-line comment
            if re.match(r'.*///.*' + re.escape(identifier), line_portion.lstrip()):
                continue # Skip if in triple-slash comment
            
            
            # Only add if it's likely a table
            if self._is_likely_table_name(identifier):
                # Extra check: must be at least 3 chars for standalone (avoid "Id" etc)
                if len(identifier) >= 3:
                    refs.add(identifier)
        
        return refs
    
    def _extract_definitions_from_lhs(self, code: str) -> Set[str]:
        """
        Extract variable/table definitions from left-hand side of assignments.
        
        Returns base names (e.g., "Items" from "Items.Field = ...")
        """
        defs = set()
        
        # Pattern: identifier = ... or Table.field = ...
        # Look at start of lines (possibly with indentation)
        for line in code.split('\n'):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
            
            # Check for assignment
            match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*=', line)
            if match:
                var_name = match.group(1)
                # Extract base name (before the dot if present)
                base_name = var_name.split('.')[0]
                defs.add(base_name)
        
        return defs
    
    def _remove_all_strings(self, code: str) -> str:
        """
        Remove all string literals from code to avoid matching their content.
        Handles single quotes, double quotes, and triple quotes.
        """
        # Remove triple-quoted strings first
        code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)
        # Remove double-quoted strings
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        # Remove single-quoted strings
        code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
        return code
    
    def _extract_path_variables(self, code: str) -> Set[str]:
        """Extract variables used in path interpolation like \\{varName}"""
        return set(re.findall(r'\\\{(\w+)\}', code))
    
    def _extract_interpolated_variables(self, code: str) -> Set[str]:
        """Extract variables from #(varName) interpolation syntax"""
        # Pattern: #(identifier) or #(Table.field)
        pattern = re.compile(r'#\(([a-zA-Z_]\w*(?:\.\w+)?)\)')
        refs = set()
        
        for match in pattern.finditer(code):
            var_ref = match.group(1)
            # Extract base name (before dot if present)
            base_name = var_ref.split('.')[0]
            # Only add if it's likely a variable (not a builtin)
            if base_name.lower() not in self.BUILTINS and len(base_name) >= 2:
                refs.add(base_name)
        
        return refs
    
    def _is_likely_table_name(self, name: str) -> bool:
        """Check if a name is likely to be a table name based on conventions."""
        # Not a builtin
        if name.lower() in self.BUILTINS:
            return False
        # Not a common property
        if name in self.COMMON_PROPERTIES:
            return False
        # Has reasonable length (not too short)
        if len(name) < 2:
            return False
        return True
    
    def _parse_block(self, lines: List[str], start_idx: int) -> Tuple[Optional[CodeBlock], int]:
        """
        Parse a single block starting at the given index.
        
        Args:
            lines: List of all lines in the script
            start_idx: Index to start parsing from
            
        Returns:
            Tuple of (CodeBlock or None, number of lines consumed)
        """
        if start_idx >= len(lines):
            return None, 0
        
        line = lines[start_idx]
        
        # Skip empty lines
        if not line.strip():
            return None, 1
        
        # Section header (special comment type)
        if self.PATTERNS['section_header'].match(line):
            return self._parse_section_header(lines, start_idx)
        
        # Form read (must be before regular read)
        if self.PATTERNS['form_read'].match(line):
            return self._parse_form_read(lines, start_idx)
        
        # Import statement
        if self.PATTERNS['import'].match(line):
            return self._parse_import(lines, start_idx)
        
        # Read statement
        if self.PATTERNS['read'].match(line):
            return self._parse_read(lines, start_idx)
        
        # Write statement
        if self.PATTERNS['write'].match(line):
            return self._parse_write(lines, start_idx)
        
        # Const declaration
        if self.PATTERNS['const'].match(line):
            return self._parse_const(lines, start_idx)
        
        # Export declaration
        if self.PATTERNS['export'].match(line):
            return self._parse_export(lines, start_idx)
        
        # Table definition
        if self.PATTERNS['table'].match(line):
            return self._parse_table(lines, start_idx)
        
        # Show statement
        if self.PATTERNS['show'].match(line):
            return self._parse_show(lines, start_idx)
        
        # Keep/Where statements
        if self.PATTERNS['keep'].match(line) or self.PATTERNS['where'].match(line):
            return self._parse_keep_where(lines, start_idx)
        
        # Comment
        if self.PATTERNS['comment'].match(line):
            return self._parse_comment(lines, start_idx)
        
        # Default: treat as assignment or unknown
        return self._parse_assignment(lines, start_idx)
    
    def _parse_section_header(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a section header comment block."""
        content_lines = []
        line_numbers = []
        i = start_idx
        
        # Collect the header line and any following comment lines
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            if self.PATTERNS['comment'].match(line):
                content_lines.append(line)
                line_numbers.append(i + 1)
                i += 1
            else:
                break
        
        # Extract section name from the header
        # Pattern: ///==== Section Name ==== or ///~~~~ Section Name ~~~~
        name = None
        if content_lines:
            for line in content_lines:
                if self.PATTERNS['section_header'].match(line):
                    # Remove comment markers and separator characters
                    cleaned = re.sub(r'^///\s*[~=\-]+\s*', '', line)
                    cleaned = re.sub(r'\s*[~=\-]+\s*/*\s*$', '', cleaned)
                    cleaned = cleaned.strip()
                    # Only use as name if there's actual text (not just separators)
                    if cleaned and not re.match(r'^[~=\-/\s]*$', cleaned):
                        name = cleaned
                        break
        
        block = CodeBlock(
            content='\n'.join(content_lines),
            block_type=BlockType.SECTION_HEADER,
            name=name,
            line_start=start_idx + 1,
            line_end=i,
            metadata={'is_delimiter': True}
        )
        
        return block, i - start_idx
    
    def _parse_comment(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a comment block."""
        content_lines = []
        line_numbers = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if self.PATTERNS['section_header'].match(line):
                break
            if self.PATTERNS['comment'].match(line):
                content_lines.append(line)
                line_numbers.append(i + 1)
                i += 1
            else:
                break
        
        block = CodeBlock(
            block_type=BlockType.COMMENT,
            content='\n'.join(content_lines),
            line_start=start_idx + 1,
            line_end=i
        )
        
        return block, i - start_idx
    
    def _parse_import(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse an import statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        
        # Extract imported items
        dependencies = set()
        definitions = set()
        name = None
        
        content = '\n'.join(content_lines)
        
        # Extract alias if present (what this import defines) - use as name
        alias_match = re.search(r'\s+as\s+(\w+)', content)
        if alias_match:
            name = alias_match.group(1)
            definitions.add(name)
        
        # Extract imported items after 'with' (also definitions in current scope)
        with_match = re.search(r'\s+with\s+(.*?)$', content, re.DOTALL)
        if with_match:
            items_text = with_match.group(1)
            # Extract identifiers, but ignore keywords
            for line in items_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    # Simple identifier on a line
                    identifier_match = re.match(r'^([a-zA-Z_]\w*)(?:\s|$)', line)
                    if identifier_match:
                        item = identifier_match.group(1)
                        if item not in self.BUILTINS:
                            definitions.add(item)
        
        # Import statements typically don't have dependencies on other tables
        
        block = CodeBlock(
            block_type=BlockType.IMPORT,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_read(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a read statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        # Extract table name (what this defines)
        definitions = set()
        dependencies = set()
        name = None
        
        # Pattern: read "path" as TableName
        as_match = re.search(r'\s+as\s+([a-zA-Z_]\w*)(?:\[|expect|\s|$)', content)
        if as_match:
            name = as_match.group(1)
            definitions.add(name)
        
        # Extract path variables (these are dependencies)
        path_vars = self._extract_path_variables(content)
        dependencies.update(path_vars)
        
        # Field definitions in 'with' clause are NOT dependencies
        # They define the schema, not reference other tables
        
        block = CodeBlock(
            block_type=BlockType.READ,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_write(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a write statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        # Extract dependencies (what's being written)
        dependencies = set()
        name = None
        
        # Pattern: write TableName into "path"
        # The table being written is a dependency
        write_match = re.search(r'write\s+([a-zA-Z_]\w*)', content)
        if write_match:
            table_name = write_match.group(1)
            name = table_name  # Use table name as block name
            if self._is_likely_table_name(table_name):
                dependencies.add(table_name)
        
        # Also check for path variables
        path_vars = self._extract_path_variables(content)
        dependencies.update(path_vars)
        
        block = CodeBlock(
            block_type=BlockType.WRITE,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies
        )
        
        return block, i - start_idx
    
    def _parse_const(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a const declaration."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        definitions = set()
        dependencies = set()
        name = None
        
        # Extract const name
        const_match = re.search(r'const\s+([a-zA-Z_]\w*)\s*=', content)
        if const_match:
            name = const_match.group(1)
            definitions.add(name)
        
        # Extract dependencies from RHS (after =)
        eq_pos = content.find('=')
        if eq_pos != -1:
            rhs = content[eq_pos + 1:]
            # Extract table references from RHS
            refs = self._extract_table_references(rhs)
            dependencies.update(refs)
            
            # Also check for path variables
            path_vars = self._extract_path_variables(rhs)
            dependencies.update(path_vars)
            
            # Also check for interpolated variables
            interpolated = self._extract_interpolated_variables(rhs)
            dependencies.update(interpolated)
        
        block = CodeBlock(
            block_type=BlockType.CONST,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_export(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse an export declaration."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        definitions = set()
        dependencies = set()
        name = None
        
        # Extract exported variable/table name
        export_match = re.search(r'export\s+(?:const\s+)?(?:table\s+)?([a-zA-Z_]\w*)', content)
        if export_match:
            name = export_match.group(1)
            definitions.add(name)
        
        # Extract dependencies from RHS if there's an assignment
        eq_pos = content.find('=')
        if eq_pos != -1:
            rhs = content[eq_pos + 1:]
            refs = self._extract_table_references(rhs)
            dependencies.update(refs)
            
            # Also extract interpolated variables
            interpolated = self._extract_interpolated_variables(rhs)
            dependencies.update(interpolated)
            
            # Remove self-reference
            if definitions:
                dependencies.discard(list(definitions)[0])
        
        block = CodeBlock(
            block_type=BlockType.EXPORT,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_table(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a table definition."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        definitions = set()
        dependencies = set()
        name = None
        
        # Extract table name
        table_match = re.search(r'table\s+([a-zA-Z_]\w*)\s*(?:\[.*?\])?\s*=', content)
        if table_match:
            name = table_match.group(1)
            definitions.add(name)
        
        # Extract dependencies from RHS
        eq_pos = content.find('=')
        if eq_pos != -1:
            rhs = content[eq_pos + 1:]
            refs = self._extract_table_references(rhs)
            dependencies.update(refs)
            
            # Also extract interpolated variables
            interpolated = self._extract_interpolated_variables(rhs)
            dependencies.update(interpolated)
            
            # Remove self-reference
            if definitions:
                dependencies.discard(list(definitions)[0])
        
        block = CodeBlock(
            block_type=BlockType.TABLE_DEFINITION,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_show(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a show statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        dependencies = set()
        name = None
        
        # Extract the title/name of the show widget
        # Pattern: show table "Title" or show linechart "Title" or show summary "Title" etc.
        title_match = re.search(r'show\s+\w+\s+"([^"]+)"', content)
        if title_match:
            name = title_match.group(1)
        
        # Show statements reference tables but don't define them
        # Extract table references, being careful about what's in strings
        refs = self._extract_table_references(content)
        dependencies.update(refs)
        
        # Also extract interpolated variables from #(...) syntax
        interpolated = self._extract_interpolated_variables(content)
        dependencies.update(interpolated)
        
        block = CodeBlock(
            block_type=BlockType.SHOW,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies
        )
        
        return block, i - start_idx
    
    def _parse_keep_where(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse keep/where statements."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        dependencies = set()
        
        # Extract table references from the condition
        refs = self._extract_table_references(content)
        dependencies.update(refs)
        
        # Also extract interpolated variables
        interpolated = self._extract_interpolated_variables(content)
        dependencies.update(interpolated)
        
        block = CodeBlock(
            block_type=BlockType.KEEP_WHERE,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies
        )
        
        return block, i - start_idx
    
    def _parse_form_read(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse a form read statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        definitions = set()
        
        # Extract field names from form read (these are variables defined)
        # Pattern: fieldName : type
        for line in content_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('read'):
                field_match = re.match(r'^([a-zA-Z_]\w*)\s*:\s*\w+', stripped)
                if field_match:
                    definitions.add(field_match.group(1))
        
        block = CodeBlock(
            block_type=BlockType.FORM_READ,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _parse_assignment(self, lines: List[str], start_idx: int) -> Tuple[CodeBlock, int]:
        """Parse an assignment statement."""
        content_lines, i = self._collect_statement(lines, start_idx)
        content = '\n'.join(content_lines)
        
        definitions = set()
        dependencies = set()
        name = None
        
        # Extract LHS (what's being defined)
        # Can be: varName = ... or Table.field = ...
        first_line = content_lines[0] if content_lines else ""
        assignment_match = self.ASSIGNMENT_PATTERN.match(first_line)
        
        if assignment_match:
            var_name = assignment_match.group(1)
            # Extract base name (before the dot if present)
            base_name = var_name.split('.')[0]
            name = base_name  # Use as block name
            definitions.add(base_name)
            
            # Extract RHS dependencies (after =)
            eq_pos = content.find('=')
            if eq_pos != -1:
                rhs = content[eq_pos + 1:]
                refs = self._extract_table_references(rhs)
                dependencies.update(refs)
                
                # Also extract interpolated variables
                interpolated = self._extract_interpolated_variables(rhs)
                dependencies.update(interpolated)
                
                # Only remove self-reference if it's a simple variable assignment
                # For Table.field = ..., keep the table as both definition and dependency
                # if it appears on the RHS (e.g., Items.x = Items.y + 1)
                if '.' not in var_name:
                    # Simple variable assignment (x = y), remove self-reference
                    dependencies.discard(base_name)
                # else: Table.field assignment - keep table in dependencies if it appears on RHS
        
        block_type = BlockType.ASSIGNMENT if assignment_match else BlockType.UNKNOWN
        
        block = CodeBlock(
            block_type=block_type,
            name=name,
            content=content,
            line_start=start_idx + 1,
            line_end=i,
            dependencies=dependencies,
            definitions=definitions
        )
        
        return block, i - start_idx
    
    def _collect_statement(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """
        Collect all lines belonging to a single statement.
        
        Statements can span multiple lines and end when:
        - A new statement starts (identified by keywords at column 0 or minimal indent)
        - An empty line is encountered after the statement completes
        - End of file
        
        Key improvement: Uses indentation to detect continuation lines.
        Lines indented more than the starting line are considered continuations.
        
        Returns:
            Tuple of (collected lines, ending index)
        """
        content_lines = []
        i = start_idx
        
        # Keywords that start a new statement
        statement_starters = ['import', 'read', 'write', 'const', 'export', 'table', 'show', 'keep', 'where']
        
        # Get base indentation level from first line
        first_line = lines[start_idx]
        base_indent = len(first_line) - len(first_line.lstrip())
        
        # Track if we're inside a multi-line structure
        unclosed_brackets = 0
        in_triple_quote = False
        triple_quote_char = None
        free_pass = False  # If true, ignore normal ending rules for one line
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip completely empty lines at the start
            if i == start_idx and not stripped:
                i += 1
                continue
            
            # Get current line's indentation
            current_indent = len(line) - len(line.lstrip()) if stripped else 0
            
            # If we're past the first line and not in a multi-line structure
            if not free_pass and i > start_idx and unclosed_brackets == 0 and not in_triple_quote:
                """# Empty line can end statement if brackets are balanced
                if not stripped:
                    break"""
                
                # New statement starts at same or lower indentation level
                if current_indent <= base_indent:
                    # Check if this is a new statement
                    if any(stripped.startswith(kw + ' ') or stripped.startswith(kw + '\t') 
                           for kw in statement_starters):
                        break
                    
                    # Section header or standalone comment starts
                    if self.PATTERNS['section_header'].match(line):
                        break
                    
                    # Assignment starts
                    if self.ASSIGNMENT_PATTERN.match(line) or self.PATTERNS['comment'].match(line):
                        break

                    if not stripped:
                        content_lines.append(line)
                        # Empty lines do not impact statements
                        i += 1
                        continue

            # Check for triple-quoted strings
            if '"""' in line and (triple_quote_char != "'" or not in_triple_quote):
                in_triple_quote = not in_triple_quote
                triple_quote_char = '"'
            elif "'''" in line and (triple_quote_char != '"' or not in_triple_quote):
                in_triple_quote = not in_triple_quote
                triple_quote_char = "'"
            
            free_pass = False
            
            content_lines.append(line)
            
            # Track bracket balance (excluding strings)
            line_no_strings = self._remove_all_strings(line)
            unclosed_brackets += line_no_strings.count('(') - line_no_strings.count(')')
            unclosed_brackets += line_no_strings.count('[') - line_no_strings.count(']')
            unclosed_brackets += line_no_strings.count('{') - line_no_strings.count('}')
            
            i += 1
            
            # If next line starts with continuation keyword, continue
            if stripped and any(stripped.startswith(kw + ' ') or stripped.startswith(kw + '\t')
                                    for kw in ['with', 'by', 'at', 'when', 'scan', 'default', 'group', 'order', 'where', 'into']):
                continue
            
            # Check if line explicitly continues
            if line.rstrip().endswith('\\'):
                free_pass = True
                continue
        
        return content_lines, i
    
    def _suggests_continuation(self, line: str) -> bool:
        """Check if a line suggests continuation (ends with 'with', comma, operator, etc.)"""
        stripped = line.rstrip()
        continuation_endings = ['with', ',', 'as', '=', 'and', 'or', '+', '-', '*', '/', 'by', 'at', 'when']
        return any(stripped.endswith(ending) for ending in continuation_endings)