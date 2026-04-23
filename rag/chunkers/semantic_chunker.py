"""
Semantic chunker for Envision DSL code blocks.

This chunker understands the semantic structure of Envision DSL and creates
meaningful chunks that respect functional boundaries and logical groupings.
"""

from typing import List, Dict, Any, Optional
import logging
import os

from rag.core.base_chunker import BaseChunker, CodeChunk
from rag.core.base_parser import CodeBlock
from utils.get_mapping import get_file_mapping

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    """
    Semantic chunker that creates meaningful chunks based on code structure.
    
    The chunker groups related code blocks together while respecting:
    - Functional boundaries (complete operations)
    - Section boundaries (major workflow sections)
    - Dependency relationships
    - Token size limits
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Get strategies from configuration
        strategies = self.config.get('strategies', {})
        self.mapping = get_file_mapping()
        self.group_by_section = strategies.get('group_by_section', True)
        self.group_related_assignments = strategies.get('group_related_assignments', True)
        self.keep_read_statements_separate = strategies.get('keep_read_statements_separate', True)
        self.include_context_comments = strategies.get('include_context_comments', True)
        
        # Priority ordering from configuration
        self.block_type_priority = self.config.get('block_priorities', {
            'comment_block': 1,
            'read_statement': 2,
            'table_definition': 3,
            'assignment': 4,
            'show_statement': 5
        })
    
    def chunk_blocks(self, code_blocks: List[CodeBlock]) -> List[CodeChunk]:
        """Chunk code blocks into semantically meaningful chunks."""
        if not code_blocks:
            return []
        
        chunks = []
        
        # Group blocks by section first if enabled
        if self.group_by_section:
            sections = self._group_by_section(code_blocks)
            for section_name, section_blocks in sections.items():
                section_chunks = self._chunk_section(section_blocks, section_name)
                chunks.extend(section_chunks)
        else:
            chunks = self._chunk_section(code_blocks, "main")
        
        # Validate and adjust chunks
        chunks = self._adjust_chunk_sizes(chunks)
        
        return chunks
    
    def _group_by_section(self, code_blocks: List[CodeBlock]) -> Dict[str, List[CodeBlock]]:
        """Group code blocks by their section."""
        sections = {}
        
        for block in code_blocks:
            section_name = block.metadata.get('section', 'main')
            if section_name not in sections:
                sections[section_name] = []
            sections[section_name].append(block)
        
        return sections
    
    def _chunk_section(self, section_blocks: List[CodeBlock], section_name: str) -> List[CodeChunk]:
        """Chunk blocks within a single section."""
        chunks = []
        
        # Sort blocks by line number to maintain order
        sorted_blocks = sorted(section_blocks, key=lambda b: b.line_start)
        
        # Group blocks by type and relationships
        block_groups = self._create_semantic_groups(sorted_blocks)
        
        # Convert groups to chunks
        for group in block_groups:
            chunk = self._create_chunk_from_group(group, section_name)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_groups(self, blocks: List[CodeBlock]) -> List[List[CodeBlock]]:
        """Create semantic groups of related blocks."""
        groups = []
        current_group = []
        current_tokens = 0
        
        i = 0
        while i < len(blocks):
            block = blocks[i]
            block_tokens = self._estimate_tokens(block.content)
            
            # Special handling for different block types
            if block.block_type == 'read_statement' and self.keep_read_statements_separate:
                # Start new group for read statements
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                
                # Read statement gets its own chunk (potentially with related comments)
                read_group = [block]
                read_tokens = block_tokens
                
                # Look for related comment blocks before/after
                if self.include_context_comments:
                    # Check for comments immediately before
                    if (i > 0 and blocks[i-1].block_type == 'comment_block' and
                        blocks[i-1] not in [b for group in groups for b in group]):
                        comment_tokens = self._estimate_tokens(blocks[i-1].content)
                        # Only add comment if it doesn't violate boundaries or size limits
                        if not self.preserve_boundaries or read_tokens + comment_tokens <= self.max_chunk_tokens:
                            read_group.insert(0, blocks[i-1])
                            read_tokens += comment_tokens
                    
                    # Check for comments immediately after
                    if (i < len(blocks) - 1 and blocks[i+1].block_type == 'comment_block'):
                        comment_tokens = self._estimate_tokens(blocks[i+1].content)
                        # Only add comment if it doesn't violate boundaries or size limits
                        if not self.preserve_boundaries or read_tokens + comment_tokens <= self.max_chunk_tokens:
                            read_group.append(blocks[i+1])
                            read_tokens += comment_tokens
                            i += 1  # Skip the comment block in next iteration
                
                groups.append(read_group)
                
            elif block.block_type == 'assignment' and self.group_related_assignments:
                # Group related assignments together
                if not current_group or self._are_assignments_related(current_group[-1], block):
                    if current_tokens + block_tokens <= self.max_chunk_tokens:
                        current_group.append(block)
                        current_tokens += block_tokens
                    else:
                        # Current group is full, check preserve_boundaries
                        if self.preserve_boundaries:
                            # Preserve assignment boundaries: complete current group
                            if current_group:
                                groups.append(current_group)
                            current_group = [block]
                            current_tokens = block_tokens
                        else:
                            # Don't preserve boundaries: try to fit or split
                            if current_group:
                                groups.append(current_group)
                            
                            # Handle large assignment blocks
                            if block_tokens > self.max_chunk_tokens:
                                split_chunks = self.chunk_single_block(block)
                                for chunk in split_chunks:
                                    groups.append(chunk.original_blocks)
                                current_group = []
                                current_tokens = 0
                            else:
                                current_group = [block]
                                current_tokens = block_tokens
                else:
                    # Not related, start new group (boundaries naturally preserved)
                    if current_group:
                        groups.append(current_group)
                    current_group = [block]
                    current_tokens = block_tokens
                    
            else:
                # Default grouping logic
                if current_tokens + block_tokens <= self.max_chunk_tokens:
                    current_group.append(block)
                    current_tokens += block_tokens
                else:
                    # Current group is full - check preserve_boundaries setting
                    if self.preserve_boundaries:
                        # Respect boundaries: complete current group and start new one
                        if current_group:
                            groups.append(current_group)
                        current_group = [block]
                        current_tokens = block_tokens
                    else:
                        # Don't preserve boundaries: try to split the block if possible
                        if current_group:
                            groups.append(current_group)
                        
                        # Check if block itself exceeds limit and needs splitting
                        if block_tokens > self.max_chunk_tokens:
                            # Split the large block
                            split_chunks = self.chunk_single_block(block)
                            for chunk in split_chunks:
                                groups.append(chunk.original_blocks)
                            current_group = []
                            current_tokens = 0
                        else:
                            current_group = [block]
                            current_tokens = block_tokens
            
            i += 1
        
        # Add remaining blocks
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _are_assignments_related(self, block1: CodeBlock, block2: CodeBlock) -> bool:
        """Check if two assignment blocks are semantically related."""
        if block1.block_type != 'assignment' or block2.block_type != 'assignment':
            return False
        
        # Check if they work on the same table/variable
        name1 = block1.name or ""
        name2 = block2.name or ""
        
        # Extract base table name (e.g., "Catalog" from "Catalog.ItemCode")
        table1 = name1.split('.')[0] if '.' in name1 else name1
        table2 = name2.split('.')[0] if '.' in name2 else name2
        
        if table1 == table2:
            return True
        
        # Check if one references the other
        if table1 in block2.content or table2 in block1.content:
            return True
        
        # Check if they have similar assignment types
        type1 = block1.metadata.get('assignment_type', '')
        type2 = block2.metadata.get('assignment_type', '')
        
        if type1 == type2 and type1 in ['aggregation', 'calculation']:
            return True
        
        return False
    
    def _create_chunk_from_group(self, blocks: List[CodeBlock], section_name: str) -> Optional[CodeChunk]:
        """Create a chunk from a group of related blocks."""
        if not blocks:
            return None
        
        # Combine content from all blocks
        content_parts = []
        total_tokens = 0
        
        # Sort blocks by line number to maintain order
        sorted_blocks = sorted(blocks, key=lambda b: b.line_start)
        
        for block in sorted_blocks:
            content_parts.append(block.content)
            total_tokens += self._estimate_tokens(block.content)
        
        combined_content = '\n\n'.join(content_parts)
        
        # Determine chunk type based on predominant block types
        chunk_type = self._determine_chunk_type(blocks)
        
        # Create contextual name
        chunk_name = self._create_chunk_name(blocks)
        
        # Add context from surrounding blocks
        context = self._generate_context(blocks, section_name)
        
        # Collect metadata
        metadata = {
            'section': section_name,
            'block_types': [b.block_type for b in blocks],
            'block_count': len(blocks),
            'primary_block_type': chunk_type,
            'semantic_group': True,
            'chunk_name': chunk_name
        }
        
        file_path = blocks[0].file_path
        # Extract original file path from the first block's file if available
        metadata['original_file_path'] = self.mapping.get(os.path.splitext(os.path.basename(file_path))[0], None)
        
        # Add specific metadata based on chunk type
        if chunk_type == 'read_statement':
            table_names = [b.metadata.get('table_name') for b in blocks if b.metadata.get('table_name')]
            if table_names:
                metadata['table_names'] = table_names
        elif chunk_type == 'assignment':
            variable_names = [b.name for b in blocks if b.name]
            if variable_names:
                metadata['variable_names'] = variable_names
        
        chunk = CodeChunk(
            content=combined_content,
            chunk_type=chunk_type,
            original_blocks=blocks,
            context=context,
            size_tokens=total_tokens,
            metadata=metadata
        )
        
        return chunk
    
    def _determine_chunk_type(self, blocks: List[CodeBlock]) -> str:
        """Determine the predominant chunk type from a group of blocks."""
        type_counts = {}
        for block in blocks:
            block_type = block.block_type
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        # Find the most common type, with priority ordering
        most_common_type = max(type_counts.keys(), 
                             key=lambda t: (type_counts[t], -self.block_type_priority.get(t, 999)))
        
        # Use semantic names for chunks
        type_mapping = {
            'read_statement': 'data_ingestion',
            'table_definition': 'table_operation',
            'assignment': 'calculation',
            'show_statement': 'visualization',
            'comment_block': 'documentation'
        }
        
        return type_mapping.get(most_common_type, most_common_type)
    
    def _create_chunk_name(self, blocks: List[CodeBlock]) -> Optional[str]:
        """Create a descriptive name for the chunk."""
        # Prioritize named blocks
        named_blocks = [b for b in blocks if b.name]
        if named_blocks:
            names = [b.name for b in named_blocks[:3]]  # Limit to first 3 names
            return ', '.join(names)
        
        # Fall back to block type description
        block_types = list(set(b.block_type for b in blocks))
        if len(block_types) == 1:
            return f"{block_types[0]}_group"
        else:
            return "mixed_operations"
    
    def _generate_context(self, blocks: List[CodeBlock], section_name: str) -> str:
        """Generate contextual information for the chunk."""
        context_parts = []
        
        # Add section context
        context_parts.append(f"Section: {section_name}")
        
        # Add operation context
        operation_types = list(set(b.metadata.get('assignment_type', b.block_type) for b in blocks))
        if operation_types:
            context_parts.append(f"Operations: {', '.join(operation_types)}")
        
        # Add table context
        tables = list(set(dep for b in blocks for dep in self._extract_table_references(b.content)))
        if tables:
            from utils.config_manager import get_config
            config = get_config()
            max_tables = config.get('embedder.text_preparation.max_table_names', 5)
            context_parts.append(f"Tables: {', '.join(tables[:max_tables])}")
        
        return ' | '.join(context_parts)
    
    def _extract_table_references(self, content: str) -> List[str]:
        """Extract table references from content."""
        import re
        # Find patterns like Table.Column
        table_refs = re.findall(r'(\w+)\.', content)
        return list(set(table_refs))
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        # Simple estimation: 1 token ≈ 4 characters for code
        return len(content) // 4
    
    def _adjust_chunk_sizes(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Adjust chunk sizes if they exceed limits."""
        adjusted_chunks = []
        
        for chunk in chunks:
            if chunk.size_tokens <= self.max_chunk_tokens:
                adjusted_chunks.append(chunk)
            else:
                # Split large chunks
                self.logger.warning(f"Chunk exceeds token limit ({chunk.size_tokens} > {self.max_chunk_tokens}), splitting...")
                split_chunks = self._split_large_chunk(chunk)
                adjusted_chunks.extend(split_chunks)
        
        return adjusted_chunks
    
    def _split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split a large chunk into smaller ones."""
        # If the chunk has multiple blocks, split by blocks
        if len(chunk.original_blocks) > 1:
            return self._split_by_blocks(chunk)
        else:
            # Single large block, use parent method
            single_block = chunk.original_blocks[0]
            return self.chunk_single_block(single_block)
    
    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split a chunk by redistributing its blocks."""
        new_chunks = []
        current_blocks = []
        current_tokens = 0
        
        for block in chunk.original_blocks:
            block_tokens = self._estimate_tokens(block.content)
            
            if current_tokens + block_tokens <= self.max_chunk_tokens:
                current_blocks.append(block)
                current_tokens += block_tokens
            else:
                # Create chunk from current blocks
                if current_blocks:
                    section_name = chunk.metadata.get('section', 'main')
                    new_chunk = self._create_chunk_from_group(current_blocks, section_name)
                    if new_chunk:
                        new_chunks.append(new_chunk)
                
                # Start new chunk
                current_blocks = [block]
                current_tokens = block_tokens
        
        # Handle remaining blocks
        if current_blocks:
            section_name = chunk.metadata.get('section', 'main')
            new_chunk = self._create_chunk_from_group(current_blocks, section_name)
            if new_chunk:
                new_chunks.append(new_chunk)
        
        return new_chunks
