#!/usr/bin/env python3
"""
Envision Script Parser and Chunker

This module provides functionality to parse Envision scripts into semantic blocks
and then chunk them into fixed-size pieces while maintaining semantic coherence
and tracking dependencies.
"""

import re
from typing import List, Dict, Any
from rag.core.base_parser import BlockType, CodeBlock
from rag.core.base_chunker import CodeChunk, BaseChunker
from rag.parsers.envision_parser import EnvisionParser
from config_manager import get_config


class EnvisionChunker(BaseChunker):
    """
    Chunks parsed code blocks into semantically coherent chunks with size limits.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunker-specific configuration options
        
        min_overlap_lines: Minimum number of lines to overlap between adjacent chunks
        """
        super().__init__(config)
        self.min_overlap_lines = self.config.get('overlap_lines', 3)
    
    def _split_large_block(self, block: CodeBlock, max_tokens: int) -> List[CodeBlock]:
        """
        Split a large block into multiple smaller blocks that fit within max_tokens.
        Sub-blocks will overlap by overlap_lines to maintain context.
        
        Args:
            block: The block to split
            max_tokens: Maximum tokens per sub-block
            
        Returns:
            List of sub-blocks with overlap
        """
        # If block fits, return as-is
        if block.get_token_count() <= max_tokens:
            return [block]
        
        # Split the block evenly with overlap
        lines = block.content.split('\n')
        total_lines = len(lines)
        
        # Calculate how many sub-blocks we need
        num_subblocks = (block.get_token_count() + max_tokens - 1) // max_tokens
        
        # Calculate lines per sub-block (without overlap initially)
        # We need to account for overlap: each sub-block after the first gets overlap lines
        # Total coverage = base_lines_per_subblock * num_subblocks + overlap * (num_subblocks - 1)
        # We want this to equal total_lines
        effective_lines_per_subblock = (total_lines - self.overlap_lines * (num_subblocks - 1)) / num_subblocks
        base_lines_per_subblock = max(1, int(effective_lines_per_subblock))
        
        sub_blocks = []
        current_start_idx = 0
        
        for i in range(num_subblocks):
            # Calculate the range for this sub-block
            if i == 0:
                # First sub-block: no overlap at start
                start_idx = 0
                end_idx = min(base_lines_per_subblock, total_lines)
            else:
                # Subsequent sub-blocks: start with overlap from previous
                start_idx = max(0, current_start_idx - self.overlap_lines)
                end_idx = min(start_idx + base_lines_per_subblock + self.overlap_lines, total_lines)
            
            # Ensure we make progress (don't get stuck)
            if i > 0 and start_idx >= current_start_idx:
                start_idx = current_start_idx
            
            # Last sub-block takes all remaining lines
            if i == num_subblocks - 1:
                end_idx = total_lines
            
            # Don't create empty sub-blocks
            if start_idx >= total_lines:
                break
            
            sub_content = '\n'.join(lines[start_idx:end_idx])
            
            # Calculate actual line numbers relative to original block
            actual_start = block.line_start + start_idx
            actual_end = block.line_start + end_idx - 1

            # Calculate how many lines are overlap from previous sub-block
            overlapping_lines = 0
            if i > 0:
                overlapping_lines = min(self.overlap_lines, current_start_idx - start_idx)
            
            sub_dependencies = set()
            sub_definitions = set()
            # For simplicity, copy dependencies and definitions from original block
            # which appear in the sub-block content
            for dep in block.dependencies:
                if re.search(r'\b' + re.escape(dep) + r'\b', sub_content):
                    sub_dependencies.add(dep)

            for defi in block.definitions:
                if re.search(r'\b' + re.escape(defi) + r'\b', sub_content):
                    sub_definitions.add(defi)
            
            sub_block = CodeBlock(
                block_type=block.block_type,
                content=sub_content,
                line_start=actual_start,
                line_end=actual_end,
                dependencies=sub_dependencies,
                definitions=sub_definitions,
                metadata={
                    **block.metadata, 
                    'is_split_part': True, 
                    'original_block_start': block.line_start,
                    'part_number': i + 1,
                    'total_parts': num_subblocks,
                    'has_split_overlap': i > 0,
                    'split_overlap_lines': overlapping_lines
                }
            )
            sub_blocks.append(sub_block)
            
            # Update current position for next iteration
            current_start_idx = end_idx
            
            # If we've covered all lines, stop
            if current_start_idx >= total_lines:
                break
        
        return sub_blocks
    
    def _get_overlap_blocks(self, blocks: List[CodeBlock], 
                           chunk_end_idx: int,
                           next_chunk_start_idx: int) -> List[CodeBlock]:
        """
        Get blocks to use as overlap between two adjacent chunks.
        
        Args:
            blocks: All blocks
            chunk_end_idx: Index of last block in current chunk
            next_chunk_start_idx: Index of first block in next chunk
            
        Returns:
            List of blocks to use as overlap
        """
        overlap_blocks = []
        total_lines = 0
        
        # Go backwards from chunk_end_idx to collect blocks until we have min_overlap_lines
        idx = chunk_end_idx
        while idx >= 0 and total_lines < self.min_overlap_lines:
            block = blocks[idx]
            if block.get_token_count() > self.max_chunk_tokens//4:
                break
            overlap_blocks.insert(0, block)
            total_lines += len(block)
            idx -= 1
            
            # Stop if we've added enough blocks
            if len(overlap_blocks) >= 3 or total_lines >= self.min_overlap_lines:
                break
        
        return overlap_blocks
    
    def chunk_blocks(self, code_blocks: List[CodeBlock], start_id: int = 0) -> List[CodeChunk]:
        """
        Chunk code blocks into semantically coherent groups.
        
        Strategy:
        1. Split oversized blocks first (with overlap between sub-blocks)
        2. Use section headers as natural delimiters
        3. Add overlap between adjacent chunks (if not separated by section header)
        4. Respect max token limit
        5. Ensure contiguity (with planned overlaps)
        6. Track current section name for each chunk
        
        Args:
            blocks: List of CodeBlock objects to chunk
            start_id: Starting ID for chunk numbering
            
        Returns:
            List of CodeChunk objects
        """
        # First pass: split any oversized blocks
        processed_blocks = []
        for block in code_blocks:
            if block.get_token_count() > self.max_chunk_tokens:
                # Split large block into smaller pieces (with internal overlap)
                sub_blocks = self._split_large_block(block, self.max_chunk_tokens)
                processed_blocks.extend(sub_blocks)
            else:
                processed_blocks.append(block)
        
        chunks = []
        current_chunk = CodeChunk(content="", chunk_id=start_id, context="code_chunk for RAG database")
        last_was_section_header = False
        current_section = None  # Track the current section name
        i = 0
        
        while i < len(processed_blocks):
            block = processed_blocks[i]
            
            # Section headers are strong delimiters - start new chunk
            if block.block_type == BlockType.SECTION_HEADER:
                if current_chunk.original_blocks:
                    chunks.append(current_chunk)
                    current_chunk = CodeChunk(content="", chunk_id=len(chunks)+start_id,
                                              context="code_chunk for RAG database")
                
                # Update current section name
                current_section = block.name
                
                # Add section header to new chunk
                current_chunk.add_block(block, self.max_chunk_tokens)
                # Set section for this chunk
                current_chunk.metadata['section'] = current_section
                last_was_section_header = True
                i += 1
                continue
            
            # Try to add block to current chunk
            if current_chunk.add_block(block, self.max_chunk_tokens):
                # Set section for this chunk if not already set
                if 'section' not in current_chunk.metadata:
                    current_chunk.metadata['section'] = current_section
                last_was_section_header = False
                i += 1
            else:
                # Block doesn't fit
                if current_chunk.original_blocks:
                    # Save current chunk
                    old_chunk = current_chunk
                    chunks.append(old_chunk)
                    
                    # Start new chunk
                    current_chunk = CodeChunk(content="", chunk_id=len(chunks)+start_id,
                                              context="code_chunk for RAG database")
                    # Set section for new chunk
                    current_chunk.metadata['section'] = current_section
                    
                    # Add overlap if:
                    # 1. Previous chunk didn't end with section header
                    # 2. Current block is not a split part that already has overlap
                    #    (split parts already have overlap with their previous part)
                    should_add_overlap = (
                        not last_was_section_header and 
                        len(old_chunk.original_blocks) > 0 and
                        not block.metadata.get('has_split_overlap', False)
                    )
                    
                    if should_add_overlap:
                        # Get overlap blocks from the end of previous chunk
                        overlap_blocks = self._get_overlap_blocks(
                            old_chunk.original_blocks,
                            len(old_chunk.original_blocks) - 1,
                            i
                        )
                        
                        # Add overlap blocks to new chunk
                        for overlap_block in overlap_blocks:
                            current_chunk.add_block(overlap_block , self.max_chunk_tokens)
                        
                        # Mark these blocks as overlap in metadata
                        if overlap_blocks:
                            current_chunk.metadata['has_overlap'] = True
                            current_chunk.metadata['overlap_with_chunk'] = old_chunk.chunk_id
                            current_chunk.metadata['overlap_blocks'] = len(overlap_blocks)
                    
                    last_was_section_header = False
                
                # Try adding the block again to the new chunk
                if not current_chunk.add_block(block, self.max_chunk_tokens):
                    # Block is too large even for empty chunk
                    # This shouldn't happen after pre-splitting, but add it anyway
                    current_chunk.add_block(block, self.max_chunk_tokens)
                i += 1
        
        # Add final chunk
        if current_chunk.original_blocks:
            # Make sure final chunk has section set
            if 'section' not in current_chunk.metadata:
                current_chunk.metadata['section'] = current_section
            chunks.append(current_chunk)
        
        # Update metadata for each chunk
        for chunk in chunks:
            self._update_chunk_metadata(chunk, chunks)
        
        return chunks
    
    def _update_chunk_metadata(self, chunk: CodeChunk, all_chunks: List[CodeChunk]):
        """
        Update metadata for a chunk, including dependencies and definitions.
        
        Args:
            chunk: The chunk to update
            all_chunks: All chunks (for computing cross-chunk dependencies)
        """
        # Calculate external dependencies (dependencies not defined in this chunk)
        external_deps = chunk.dependencies - chunk.definitions
        
        # Find which chunks provide these dependencies
        providers = {}
        for dep in external_deps:
            for other_chunk in all_chunks:
                if dep in other_chunk.definitions:
                    providers[dep] = other_chunk.chunk_id
                    break
        
        chunk.metadata.update({
            'external_dependencies': external_deps,
            'dependency_providers': providers,
            'token_count': chunk.get_token_count(),
            'block_types': [block.block_type.value for block in chunk.original_blocks],
            'line_range': chunk.get_line_range()
        })


def print_chunk_summary(chunks: List[CodeChunk]):
    """
    Print a summary of chunks for debugging/analysis.
    
    Args:
        chunks: List of CodeChunk objects
    """
    print(f"\n{'='*80}")
    print(f"CHUNK SUMMARY - Total Chunks: {len(chunks)}")
    print(f"{'='*80}\n")
    
    for chunk in chunks:
        line_range = chunk.get_line_range()
        print(f"Chunk {chunk.chunk_id}:")
        print(f"  Lines: {line_range[0]}-{line_range[1]} ({len(chunk.original_blocks)} blocks)")
        print(f"  Tokens: {chunk.metadata.get('token_count', 0)}")
        print(f"  Definitions: {', '.join(sorted(chunk.definitions)) if chunk.definitions else 'None'}")
        print(f"  External Dependencies: {', '.join(sorted(chunk.metadata.get('external_dependencies', set()))) if chunk.metadata.get('external_dependencies') else 'None'}")
        print(f"  Block Types: {', '.join(chunk.metadata.get('block_types', []))}")
        print()

def parse_and_chunk_file(file_path: str):
    parser = EnvisionParser()
    chunker = EnvisionChunker(get_config())
    code_blocks = parser.parse_file(file_path)
    chunks = chunker.chunk_blocks(code_blocks)
    return code_blocks, chunks