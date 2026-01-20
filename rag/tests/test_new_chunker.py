#!/usr/bin/env python3
"""
Example: Using Envision Chunker for RAG Applications

This script demonstrates how to use the chunker output in a RAG system
for semantic search and code understanding.
"""

import json
import sys
from pathlib import Path
from typing import List, Set, Dict
sys.path.insert(0, '/home/claude')

from rag.chunkers.envision_chunker import parse_and_chunk_file, CodeChunk
from rag.parsers.envision_parser import BlockType


class EnvisionRAGHelper:
    """
    Helper class for RAG applications using chunked Envision scripts.
    """
    
    def __init__(self, chunks: List[CodeChunk]):
        """
        Initialize with parsed chunks.
        
        Args:
            chunks: List of CodeChunk objects
        """
        self.chunks = chunks
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for fast lookup."""
        # Index: definition name → chunk IDs that define it
        self.definition_index: Dict[str, Set[int]] = {}
        
        # Index: chunk ID → chunks that depend on it
        self.dependent_index: Dict[int, Set[int]] = {}
        
        # Index: block type → chunk IDs
        self.type_index: Dict[str, Set[int]] = {}
        
        for chunk in self.chunks:
            chunk_id = chunk.chunk_id
            
            # Build definition index
            for definition in chunk.definitions:
                if definition not in self.definition_index:
                    self.definition_index[definition] = set()
                self.definition_index[definition].add(chunk_id)
            
            # Build dependent index
            providers = chunk.metadata.get('dependency_providers', {})
            for dep, provider_id in providers.items():
                if provider_id not in self.dependent_index:
                    self.dependent_index[provider_id] = set()
                self.dependent_index[provider_id].add(chunk_id)
            
            # Build type index
            for block_type in chunk.metadata.get('block_types', []):
                if block_type not in self.type_index:
                    self.type_index[block_type] = set()
                self.type_index[block_type].add(chunk_id)
    
    def find_definition(self, var_name: str) -> List[CodeChunk]:
        """
        Find all chunks that define a variable.
        
        Args:
            var_name: Variable/table name to search for
            
        Returns:
            List of chunks that define this variable
        """
        chunk_ids = self.definition_index.get(var_name, set())
        return [self.chunks[cid] for cid in sorted(chunk_ids)]
    
    def get_dependencies(self, chunk_id: int, recursive: bool = False) -> Set[int]:
        """
        Get all chunks that a given chunk depends on.
        
        Args:
            chunk_id: ID of the chunk
            recursive: If True, get transitive dependencies
            
        Returns:
            Set of chunk IDs
        """
        chunk = self.chunks[chunk_id]
        providers = chunk.metadata.get('dependency_providers', {})
        deps = set(providers.values())
        
        if recursive:
            # Recursively get dependencies of dependencies
            all_deps = set(deps)
            for dep_id in deps:
                if dep_id != chunk_id:  # Avoid infinite loops
                    all_deps.update(self.get_dependencies(dep_id, recursive=True))
            return all_deps
        
        return deps
    
    def get_dependents(self, chunk_id: int, recursive: bool = False) -> Set[int]:
        """
        Get all chunks that depend on a given chunk.
        
        Args:
            chunk_id: ID of the chunk
            recursive: If True, get transitive dependents
            
        Returns:
            Set of chunk IDs
        """
        deps = self.dependent_index.get(chunk_id, set())
        
        if recursive:
            all_deps = set(deps)
            for dep_id in deps:
                if dep_id != chunk_id:
                    all_deps.update(self.get_dependents(dep_id, recursive=True))
            return all_deps
        
        return deps
    
    def get_context_for_chunk(self, chunk_id: int, 
                             include_dependencies: bool = True,
                             include_dependents: bool = False) -> List[CodeChunk]:
        """
        Get contextual chunks for a given chunk.
        
        This is useful for RAG: when retrieving a chunk, also get its context.
        
        Args:
            chunk_id: ID of the target chunk
            include_dependencies: Include chunks this one depends on
            include_dependents: Include chunks that depend on this one
            
        Returns:
            List of contextual chunks (sorted by chunk ID)
        """
        context_ids = {chunk_id}
        
        if include_dependencies:
            context_ids.update(self.get_dependencies(chunk_id, recursive=True))
        
        if include_dependents:
            context_ids.update(self.get_dependents(chunk_id, recursive=True))
        
        return [self.chunks[cid] for cid in sorted(context_ids)]
    
    def find_by_type(self, block_type: str) -> List[CodeChunk]:
        """
        Find all chunks containing a specific block type.
        
        Args:
            block_type: Type of block to search for (e.g., 'read', 'show')
            
        Returns:
            List of matching chunks
        """
        chunk_ids = self.type_index.get(block_type, set())
        return [self.chunks[cid] for cid in sorted(chunk_ids)]
    
    def search_content(self, keyword: str, case_sensitive: bool = False) -> List[CodeChunk]:
        """
        Simple keyword search in chunk content.
        
        Args:
            keyword: Keyword to search for
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching chunks
        """
        results = []
        for chunk in self.chunks:
            content = chunk.get_content()
            if not case_sensitive:
                content = content.lower()
                keyword = keyword.lower()
            
            if keyword in content:
                results.append(chunk)
        
        return results
    
    def export_for_vectordb(self, output_path: str):
        """
        Export chunks in a format suitable for vector databases.
        
        Args:
            output_path: Path to save JSON file
        """
        data = []
        
        for chunk in self.chunks:
            entry = {
                'id': f'chunk_{chunk.chunk_id}',
                'content': chunk.get_content(),
                'metadata': {
                    'chunk_id': chunk.chunk_id,
                    'line_range': list(chunk.get_line_range()),
                    'token_count': chunk.get_token_count(),
                    'definitions': list(chunk.definitions),
                    'dependencies': list(chunk.metadata.get('external_dependencies', set())),
                    'dependency_providers': chunk.metadata.get('dependency_providers', {}),
                    'block_types': chunk.metadata.get('block_types', [])
                }
            }
            data.append(entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(data)} chunks to {output_path}")


def demonstrate_rag_usage():
    """Demonstrate RAG helper functionality."""
    
    # Parse a file
    print("="*80)
    print("ENVISION RAG HELPER DEMONSTRATION")
    print("="*80 + "\n")
    
    # Use one of the test files
    test_file = "env_scripts/68006.nvn"
    print(f"Parsing: {Path(test_file).name}\n")
    
    blocks, chunks = parse_and_chunk_file(test_file, max_tokens=512)
    
    # Initialize RAG helper
    rag = EnvisionRAGHelper(chunks)
    
    print(f"Total chunks: {len(chunks)}\n")
    
    # Example 1: Find where a table is defined
    print("-" * 80)
    print("Example 1: Find definition of 'Items'")
    print("-" * 80)
    items_chunks = rag.find_definition('Items')
    if items_chunks:
        for chunk in items_chunks:
            print(f"  Found in chunk {chunk.chunk_id} (lines {chunk.get_line_range()})")
            print(f"  Also defines: {', '.join(sorted(chunk.definitions)[:5])}")
    print()
    
    # Example 2: Get dependencies for a chunk
    print("-" * 80)
    print("Example 2: Get dependencies for chunk 2")
    print("-" * 80)
    if len(chunks) > 2:
        deps = rag.get_dependencies(2, recursive=True)
        print(f"  Chunk 2 depends on chunks: {sorted(deps)}")
        
        # Show what each dependency provides
        chunk2 = chunks[2]
        providers = chunk2.metadata.get('dependency_providers', {})
        if providers:
            print(f"  Specific dependencies:")
            for var, provider_id in sorted(providers.items())[:5]:
                print(f"    - {var} from chunk {provider_id}")
    print()
    
    # Example 3: Find all visualization chunks
    print("-" * 80)
    print("Example 3: Find all chunks with 'show' statements")
    print("-" * 80)
    show_chunks = rag.find_by_type('show')
    print(f"  Found {len(show_chunks)} chunks with visualizations")
    for chunk in show_chunks[:3]:
        line_range = chunk.get_line_range()
        print(f"    - Chunk {chunk.chunk_id} (lines {line_range[0]}-{line_range[1]})")
    print()
    
    # Example 4: Get context for a chunk (for RAG retrieval)
    print("-" * 80)
    print("Example 4: Get full context for chunk 2 (for RAG)")
    print("-" * 80)
    if len(chunks) > 2:
        context = rag.get_context_for_chunk(2, include_dependencies=True)
        print(f"  Context includes {len(context)} chunks: {[c.chunk_id for c in context]}")
        print(f"  Total context size: {sum(c.get_token_count() for c in context)} tokens")
        
        # Show what each context chunk provides
        print(f"  Context breakdown:")
        for ctx_chunk in context:
            if ctx_chunk.definitions:
                defs = ', '.join(sorted(ctx_chunk.definitions)[:3])
                if len(ctx_chunk.definitions) > 3:
                    defs += f" ... (+{len(ctx_chunk.definitions)-3} more)"
                print(f"    - Chunk {ctx_chunk.chunk_id}: defines {defs}")
    print()
    
    # Example 5: Search for keywords
    print("-" * 80)
    print("Example 5: Search for 'Orders' in content")
    print("-" * 80)
    results = rag.search_content('Orders')
    print(f"  Found in {len(results)} chunks:")
    for chunk in results[:3]:
        print(f"    - Chunk {chunk.chunk_id}")
    print()
    
    # Export for vector database
    print("-" * 80)
    print("Example 6: Export for vector database")
    print("-" * 80)
    output_file = "data/chunker_test/rag_export.json"
    rag.export_for_vectordb(output_file)
    print()
    
    # Show example of exported structure
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print("Example exported entry:")
    print(json.dumps(data[0], indent=2)[:500] + "...")
    print()
    
    print("="*80)
    print("RAG HELPER DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_file = "env_scripts/67984.nvn"
    print(f"Parsing: {Path(test_file).name}\n")
    
    blocks, chunks = parse_and_chunk_file(test_file)
    
    for block in blocks:
        if block.block_type != BlockType.SECTION_HEADER:
            continue  # Skip section headers for this display
        print(f"Block Type: {block.block_type.value}, Name :{block.name}, Tokens: {block.get_token_count()}")
        print(f"Line_start: {block.line_start}, Line_end: {block.line_end}")
        print(f"Definitions: {block.definitions}, Dependencies: {block.dependencies}")
        print(f"Content:\n{block.content}\n{'-'*40}\n")