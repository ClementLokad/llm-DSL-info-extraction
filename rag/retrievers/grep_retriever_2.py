import re
import pickle
from typing import List, Optional
import numpy as np
from pathlib import Path

from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_chunker import CodeChunk


class GrepRetriever2(BaseRetriever):
    """
    A retriever that ignores embeddings and performs pure grep-like
    regex-based matching on stored CodeChunks.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._embedding_dim = None

    def initialize(self, embedding_dimension: int) -> None:
        """
        Required by BaseRetriever but unused for grep.
        """
        self._embedding_dim = embedding_dimension
        self._is_initialized = True

    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """
        Store chunks normally. Embeddings are ignored.
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks count and embeddings rows mismatch")

        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        file_path: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Here query_embedding is misused: it must contain the search pattern as a string.
        This allows full compatibility with BaseRetriever.

        Convention:
            - query_embedding is a numpy array of shape (1,) containing a string
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("query_embedding must contain a grep pattern string")

        # Extract pattern
        pattern = query_embedding[0]
        if not isinstance(pattern, str):
            raise ValueError("For grep retriever, query_embedding[0] must be a string pattern")

        regex = re.compile(pattern, re.IGNORECASE)

        results = []
        rank = 1

        for chunk in self._chunks:
            # Filter by file_path if provided
            if file_path is not None and getattr(chunk, "file_path", None) != file_path:
                continue

            # Match on the full chunk content
            if regex.search(chunk.content):
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=1.0,            # constant score (grep has no similarity metric)
                        rank=rank,
                        metadata={"pattern": pattern, "file_path": getattr(chunk, "file_path", None)}
                    )
                )
                rank += 1

        # top_k is respected if requested
        top_k = top_k or self.default_top_k
        return results[:top_k]

    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save chunks only (embeddings irrelevant).
        """
        path = path or self.index_path
        try:
            with open(path, "wb") as f:
                pickle.dump(self._chunks, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save grep index: {e}")

    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load chunks from disk.
        """
        path = path or self.index_path
        try:
            with open(path, "rb") as f:
                self._chunks = pickle.load(f)
            self._is_initialized = True
        except FileNotFoundError:
            raise FileNotFoundError(f"Grep index not found at {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load grep index: {e}")
