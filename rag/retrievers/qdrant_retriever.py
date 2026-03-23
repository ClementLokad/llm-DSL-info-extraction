import uuid
import numpy as np
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from rag.embedders.qdrant_embedder import QdrantEmbedder

from rag.core.base_retriever import BaseRetriever, RetrievalResult
from rag.core.base_chunker import CodeChunk

from config_manager import get_config

class QdrantRetriever(BaseRetriever):
    """
    Retriever implementation using Qdrant Vector Database.
    
    Supports both standard dense retrieval (backward compatible with FAISS) 
    and SOTA Hybrid Retrieval (Dense + Sparse/BM25) with Reciprocal Rank Fusion.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.collection_name = self.config.get('collection_name', 'codebase_rag')
        self.rerank_multiplier = self.config.get('rerank_multiplier', 2)
        self.index_type = get_config().get("embedder.index_type", "full_chunk")
        
        # Initialize Qdrant client (Use path for local SQLite-like storage, or url/api_key for cloud)
        qdrant_path = self.config.get('qdrant_path', self.index_path)
        self.client = QdrantClient(path=qdrant_path)
        
        # Internal mapping to quickly reconstruct chunks from Qdrant IDs
        self._chunk_map: Dict[str, CodeChunk] = {}

    def initialize(self, embedding_dimension: int) -> None:
        """
        Creates the Qdrant collection with both Dense and Sparse vector spaces.
        """
        for collection in (self.collection_name, f"{self.collection_name}_summary"):
            if not self.client.collection_exists(collection):
                self.logger.info(f"Creating new Qdrant collection: {collection}")
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=embedding_dimension,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    }
                )
                
                # Only create payload indexes if NOT in local mode to avoid warnings
                # Note: Checking for 'path' in config is a simple way to detect local mode
                if not self.config.get('local'):
                    self.client.create_payload_index(
                        collection_name=collection,
                        field_name="original_file_path",
                        field_schema=models.TextIndexParams(
                        type="text",
                            tokenizer=models.TokenizerType.PREFIX,
                            min_token_len=2,
                            max_token_len=50,
                            lowercase=True,
                        )
                    )
                else:
                    self.logger.info("Local mode detected: Skipping explicit payload indexing.")
            else:
                self.logger.info(f"Connected to existing Qdrant collection: {collection}")
            
        self._is_initialized = True

    # ---------------------------------------------------------
    # BACKWARD COMPATIBLE METHODS (DENSE ONLY)
    # ---------------------------------------------------------

    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray, summary: bool = False) -> None:
        """
        Standard dense-only ingestion (backward compatible).
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")
        
        collection = f"{self.collection_name}_summary" if summary else self.collection_name
            
        points = []
        for i, chunk in enumerate(chunks):
            # Ensure chunk has an ID, fallback to UUID if necessary
            point_id = chunk.chunk_id if chunk.chunk_id is not None else str(uuid.uuid4())
            
            # Store in local memory and base class list
            self._chunks.append(chunk)
            self._chunk_map[point_id] = chunk
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": embeddings[i].tolist()},
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type,
                        "original_blocks": chunk.original_blocks,
                        "context": chunk.context,
                        "size_tokens": chunk.size_tokens,
                        "dependencies": list(chunk.dependencies) if chunk.dependencies else [],
                        "definitions": list(chunk.definitions) if chunk.definitions else [],
                        "metadata": chunk.metadata
                    }
                )
            )
            
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Standard dense-only search.
        """
        limit = top_k or self.default_top_k
        
        collection = self.collection_name if self.index_type == "full_chunk" else f"{self.collection_name}_summary"
        
        search_result = self.client.query_points(
            collection_name=collection,
            query=query_embedding.tolist(),
            using="dense",
            limit=limit
        )
        
        return self._format_results(search_result.points)

    # ---------------------------------------------------------
    # SOTA HYBRID METHODS
    # ---------------------------------------------------------

    def add_chunks_hybrid(self, chunks: List[CodeChunk], hybrid_embeddings: List[Dict[str, Any]], summary: bool = False) -> None:
        """
        Ingests chunks with both dense semantic vectors and sparse BM25 vectors.
        """
        if not self._is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        collection = f"{self.collection_name}_summary" if summary else self.collection_name

        points = []
        for chunk, embeddings in zip(chunks, hybrid_embeddings):
            point_id = chunk.chunk_id if chunk.chunk_id is not None else str(uuid.uuid4())
            
            self._chunks.append(chunk)
            self._chunk_map[point_id] = chunk
            
            # Extract fastembed sparse properties
            sparse_vec = embeddings["sparse"]
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embeddings["dense"],
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist()
                        )
                    },
                    
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type,
                        "original_blocks": chunk.original_blocks,
                        "context": chunk.context,
                        "size_tokens": chunk.size_tokens,
                        "dependencies": list(chunk.dependencies) if chunk.dependencies else [],
                        "definitions": list(chunk.definitions) if chunk.definitions else [],
                        "metadata": chunk.metadata,
                        "original_file_path": chunk.metadata.get("original_file_path", "") # Store original file path for potential source-based filtering
                    }
                )
            )
            
        self.client.upsert(collection_name=collection, points=points)

    def search_hybrid(
        self, 
        query_text: str, 
        embedder: QdrantEmbedder, 
        top_k: Optional[int] = None,
        source_substrings: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF) and optional metadata filtering.
        """
        limit = top_k or self.default_top_k
        hybrid_query = embedder.embed_text_hybrid(query_text, keywords or [])
        sparse_vec = hybrid_query["sparse"]
        collection = self.collection_name if self.index_type == "full_chunk" else f"{self.collection_name}_summary"
        
        # Stream 1 & 2: Independent Global Searches (No Filters)
        prefetch_queries = [
            models.Prefetch(
                query=hybrid_query["dense"],
                using="dense",
                limit=limit * self.rerank_multiplier,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                ),
                using="sparse",
                limit=limit * self.rerank_multiplier,
            )
        ]

        # Stream 3 & 4: Independent Source Boosting for BOTH Dense and Sparse
        if source_substrings and source_substrings != []:
            source_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="original_file_path",
                        match=models.MatchText(text=substring)
                    ) for substring in source_substrings
                ]
            )
            
            # Stream 3: Filtered Dense (For conceptual source boosting)
            prefetch_queries.append(
                models.Prefetch(
                    query=hybrid_query["dense"], 
                    using="dense",
                    filter=source_filter,
                    limit=limit * self.rerank_multiplier,
                    score_threshold=self.config.get("dense_threshold", 0.45)
                )
            )
            
            # Stream 4: Filtered Sparse (For exact keyword source boosting)
            prefetch_queries.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    ),
                    using="sparse",
                    filter=source_filter,
                    limit=limit * self.rerank_multiplier,
                    score_threshold=self.config.get("sparse_threshold", 2)
                )
            )

        # Execute Multi-Stage Prefetch Query with RRF
        search_result = self.client.query_points(
            collection_name=collection,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )

        return self._format_results(search_result.points)
    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    def _format_results(self, qdrant_points: List[Any]) -> List[RetrievalResult]:
        """Maps Qdrant's scored points back to your RetrievalResult schema."""
        results = []
        for rank, point in enumerate(qdrant_points, start=1):
            chunk = self._chunk_map.get(point.id)
            if not chunk:
                # Fallback: reconstruct chunk from payload if not in memory (e.g., after restart)
                chunk = CodeChunk(
                    content=point.payload.get("content", ""),
                    chunk_id=point.payload.get("chunk_id", point.id),
                    chunk_type=point.payload.get("chunk_type", "unknown"),
                    original_blocks=point.payload.get("original_blocks", []),
                    context=point.payload.get("context", {}),
                    size_tokens=point.payload.get("size_tokens", 0),
                    dependencies=set(point.payload.get("dependencies", [])),
                    definitions=set(point.payload.get("definitions", [])),
                    metadata=point.payload.get("metadata", {})
                )
                
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=point.score,
                    rank=rank,
                    metadata={"qdrant_id": point.id, "original_file_path": chunk.metadata.get("original_file_path", "")}
                )
            )
                
        return results

    def save_index(self, path: Optional[str] = None) -> None:
        """Qdrant handles disk persistence automatically; no-op required."""
        self.logger.info("Qdrant index is automatically persisted to disk.")
        pass

    def load_index(self, path: Optional[str] = None) -> None:
        """Qdrant loads from disk upon client initialization; no-op required."""
        self.logger.info("Qdrant index is automatically loaded upon initialization.")
        pass
        
    def clear_index(self, summary: bool = False) -> None:
        """Deletes the collection and clears memory."""
        collection = f"{self.collection_name}_summary" if summary else self.collection_name
        self.client.delete_collection(collection)
        self._chunks = []
        self._chunk_map = {}
        self._is_initialized = False
    
    def close(self):
        """Explicitly close the Qdrant client to ensure data is flushed to disk."""
        if hasattr(self, 'client') and self.client:
            self.logger.info("Closing Qdrant client...")
            self.client.close()