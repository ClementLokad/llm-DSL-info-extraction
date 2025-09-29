"""
Complete preprocessing pipeline for Envision DSL files.
Embedders are now in the agents/ directory for modular architecture.
"""
import os
from pathlib import Path
import pickle
from typing import List, Dict, Any, Optional
from agents.base import BaseEmbedder

class EnvisionProcessor:
    """Processor for Envision DSL code."""
    
    def __init__(self):
        self.comment_single = "//"
        self.comment_multi_start = "/*"
        self.comment_multi_end = "*/"
    
    def clean_code(self, content: str) -> str:
        """Clean and normalize Envision DSL code."""
        lines = content.split('\n')
        cleaned_lines = []
        in_multiline_comment = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle multi-line comments
            if self.comment_multi_start in line and not in_multiline_comment:
                if self.comment_multi_end in line:
                    # Single line /* ... */ comment
                    start = line.find(self.comment_multi_start)
                    end = line.find(self.comment_multi_end) + len(self.comment_multi_end)
                    line = line[:start] + line[end:]
                else:
                    in_multiline_comment = True
                    line = line[:line.find(self.comment_multi_start)]
            elif in_multiline_comment:
                if self.comment_multi_end in line:
                    in_multiline_comment = False
                    line = line[line.find(self.comment_multi_end) + len(self.comment_multi_end):]
                else:
                    continue
            
            # Remove single line comments
            if self.comment_single in line:
                line = line[:line.find(self.comment_single)]
            
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class EnvisionChunker:
    """Intelligent chunker for Envision DSL code."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.processor = EnvisionProcessor()
    
    def chunk_text(self, text: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Split text into semantic chunks with metadata."""
        cleaned_text = self.processor.clean_code(text)
        
        if len(cleaned_text) <= self.chunk_size:
            return [{
                'content': cleaned_text,
                'file_path': file_path,
                'start_pos': 0,
                'end_pos': len(cleaned_text),
                'metadata': self._extract_metadata(cleaned_text)
            }]
        
        chunks = []
        start = 0
        
        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # Try to break at sensible boundaries
            if end < len(cleaned_text):
                # Look for good break points (newlines, operators, etc.)
                for break_char in ['\n', ';', ')', '}', ' ']:
                    break_pos = cleaned_text.rfind(break_char, start, end)
                    if break_pos > start + self.chunk_size // 2:
                        end = break_pos + 1
                        break
            
            chunk_content = cleaned_text[start:end].strip()
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'file_path': file_path,
                    'start_pos': start,
                    'end_pos': end,
                    'metadata': self._extract_metadata(chunk_content)
                })
            
            start = max(start + 1, end - self.overlap)
        
        return chunks
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from code chunk."""
        metadata = {
            'contains_read': 'read ' in text.lower(),
            'contains_const': 'const ' in text.lower(),
            'contains_export': 'export ' in text.lower(),
            'contains_table': 'table ' in text.lower(),
            'contains_calculation': any(op in text for op in ['=', '+', '-', '*', '/', '%']),
            'line_count': text.count('\n') + 1,
            'char_count': len(text)
        }
        return metadata

class VectorSearch:
    """FAISS-based vector similarity search."""
    
    def __init__(self, dimension: int):
        try:
            import faiss
            self.faiss = faiss
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.chunks = []
        except ImportError:
            print("Warning: FAISS not available. Search functionality disabled.")
            self.faiss = None
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks and their embeddings to the search index."""
        if not self.faiss:
            return
        
        import numpy as np
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if not self.faiss or len(self.chunks) == 0:
            return []
        
        import numpy as np
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        query_norm = np.linalg.norm(query_array)
        if query_norm > 0:
            query_array = query_array / query_norm
        
        scores, indices = self.index.search(query_array, min(k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results

class PreprocessingPipeline:
    """Complete preprocessing pipeline."""
    
    def __init__(self, embedder: BaseEmbedder, chunk_size: int = 512, overlap: int = 50):
        self.embedder = embedder
        self.chunker = EnvisionChunker(chunk_size, overlap)
        self.search = VectorSearch(embedder.dimension)
    
    def process_files(self, directory: str, pattern: str = "*.nvn") -> Dict[str, Any]:
        """Process all files in directory matching pattern."""
        dir_path = Path(directory)
        files = list(dir_path.glob(pattern))
        
        if not files:
            raise ValueError(f"No files found matching pattern '{pattern}' in '{directory}'")
        
        print(f"Found {len(files)} files to process...")
        
        all_chunks = []
        file_stats = {}
        
        for file_path in files:
            print(f"Processing {file_path.name}...")
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                chunks = self.chunker.chunk_text(content, str(file_path))
                
                all_chunks.extend(chunks)
                file_stats[str(file_path)] = {
                    'chunk_count': len(chunks),
                    'char_count': len(content),
                    'line_count': content.count('\n') + 1
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Generated {len(all_chunks)} chunks from {len(files)} files")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Add to search index
        self.search.add_chunks(all_chunks, embeddings)
        
        return {
            'chunks': all_chunks,
            'embeddings': embeddings,
            'file_stats': file_stats,
            'total_chunks': len(all_chunks),
            'num_files': len(files),
            'embedder_type': type(self.embedder).__name__,
            'embedder_dimension': self.embedder.dimension
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file (excluding search index due to pickle issues)."""
        # Create a copy without the search engine (can't be pickled)
        results_to_save = results.copy()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"Results saved to {output_path}")