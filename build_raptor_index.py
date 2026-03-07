#!/usr/bin/env python3
"""Build search index with RAPTOR using pre-computed base summaries"""
import sys
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.summarizers.chunk_summarizer import ChunkSummarizer
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever
from rag.core.base_chunker import CodeChunk 


class RaptorClusterer:
    """Handles soft clustering for RAPTOR architecture using UMAP and GMM."""
    
    def __init__(self, max_clusters: int = 50, proba_threshold: float = 0.2):
        self.max_clusters = max_clusters
        self.proba_threshold = proba_threshold

    def cluster_embeddings(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        n_samples = len(embeddings)
        
        if n_samples < 5:
            return {0: list(range(n_samples))}

        n_neighbors = min(15, n_samples - 1)
        n_components = min(10, n_samples - 2)
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=n_components, 
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        max_k = min(self.max_clusters, n_samples // 2)
        max_k = max(2, max_k)

        bics = []
        gmms = []
        
        for k in range(1, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(reduced_embeddings)
            bics.append(gmm.bic(reduced_embeddings))
            gmms.append(gmm)

        optimal_k_index = np.argmin(bics)
        best_gmm = gmms[optimal_k_index]
        
        print(f"  🔍 GMM found {best_gmm.n_components} optimal clusters (BIC: {bics[optimal_k_index]:.2f})")

        probs = best_gmm.predict_proba(reduced_embeddings)
        clusters = {i: [] for i in range(best_gmm.n_components)}

        for doc_idx, doc_probs in enumerate(probs):
            assigned = False
            for cluster_idx, prob in enumerate(doc_probs):
                if prob > self.proba_threshold:
                    clusters[cluster_idx].append(doc_idx)
                    assigned = True
                    
            if not assigned:
                best_cluster = int(np.argmax(doc_probs))
                clusters[best_cluster].append(doc_idx)

        return clusters


def build_index():
    print("🔨 Building RAPTOR index from pre-computed base summaries...")
    cfg = ConfigManager()
    
    base_summarizer = ChunkSummarizer(cfg.get_summarizer_config(), raptor=False)
    raptor_summarizer = ChunkSummarizer(cfg.get_summarizer_config(), raptor=True)
    
    env_parser = EnvisionParser(cfg.get_parser_config())
    chunker = EnvisionChunker(cfg.get_chunker_config())
    embedder = SentenceTransformerEmbedder(cfg.get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(cfg.get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    dirs = cfg.get('paths.input_dirs', ["env_scripts"])
    blocks = []
    chunks: List[CodeChunk] = []
    
    # Level 0: Parsing and Chunking
    print("Reading source files...")
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.glob("*.nvn"):
            current_blocks = env_parser.parse_file(str(f))
            blocks.extend(current_blocks)
            current_chunks = chunker.chunk_blocks(current_blocks)
            
            for c in current_chunks:
                if 'original_file_path' not in c.metadata:
                    c.metadata['original_file_path'] = str(f)
                    
            chunks.extend(current_chunks)
    
    print(f"Parsed {len(blocks)} blocks and created {len(chunks)} base chunks (Level 0)")

    # Level 0: Load existing summaries and embed
    base_summaries = base_summarizer.get_summary_list()
    
    if len(base_summaries) != len(chunks):
        print(f"\n❌ CRITICAL ERROR: Data desynchronization.")
        print(f"Generated {len(chunks)} chunks, but found {len(base_summaries)} summaries in cache.")
        print("Please run 'build_summary_index.py' first to update base summaries.")
        sys.exit(1)

    print(f"✅ Loaded {len(base_summaries)} Level 0 summaries from cache.")
    
    for i, summary in enumerate(base_summaries):
        chunks[i].metadata['summary'] = summary
        chunks[i].metadata['raptor_level'] = 0

    print("Generating embeddings for Level 0...")
    base_embeddings = embedder.embed_chunks(chunks)

    # Levels 1 to N: RAPTOR Recursive Clustering
    print("\n🌳 Starting RAPTOR tree construction...")
    clusterer = RaptorClusterer(max_clusters=10, proba_threshold=0.2)
    
    all_chunks = chunks.copy()
    all_embeddings = [base_embeddings] 
    
    current_level_chunks = chunks.copy()
    current_embeddings = base_embeddings
    
    level = 1
    max_levels = 5 
    
    while len(current_level_chunks) > 1 and level <= max_levels:
        print(f"\n--- 🔼 Creating Level {level} ---")
        print(f"Documents to cluster: {len(current_level_chunks)}")
        
        clusters_dict = clusterer.cluster_embeddings(current_embeddings)
        
        if len(clusters_dict) <= 1 and level > 1:
            print("🏁 Reached tree root (single cluster).")
            break
            
        next_level_chunks = []
        total_clusters = len(clusters_dict)
        
        for idx, (cluster_id, chunk_indices) in enumerate(clusters_dict.items()):
            print(f"  ⏳ Generating summary for cluster {idx + 1}/{total_clusters}...", end='\r')
            
            child_chunks = [current_level_chunks[i] for i in chunk_indices]
            cluster_text = "\n\n".join([c.content for c in child_chunks])
            
            raw_summary = raptor_summarizer.summarize_text(cluster_text)
            
            final_summary = raw_summary
            cluster_file_paths = set()
            
            if level == 1:
                for child in child_chunks:
                    path = child.metadata.get('original_file_path')
                    if path:
                        cluster_file_paths.add(path)
                
                if cluster_file_paths:
                    prefix = "Summary of chunks from the following files:\n"
                    for path in sorted(list(cluster_file_paths)):
                        prefix += f"- {path}\n"
                    prefix += "\n"
                    final_summary = prefix + raw_summary

            parent_metadata = {
                "raptor_level": level,
                "cluster_id": cluster_id,
                "summary": final_summary
            }
            
            if cluster_file_paths:
                parent_metadata['original_file_path'] = list(cluster_file_paths)
            elif level > 1:
                inherited_paths = set()
                for child in child_chunks:
                    paths = child.metadata.get('original_file_path', [])
                    if isinstance(paths, list):
                        inherited_paths.update(paths)
                    elif isinstance(paths, str):
                        inherited_paths.add(paths)
                if inherited_paths:
                    parent_metadata['original_file_path'] = list(inherited_paths)

            parent_chunk = CodeChunk(
                content=final_summary,
                chunk_type="raptor_summary_node",
                metadata=parent_metadata
            )
            next_level_chunks.append(parent_chunk)
            
        print(f"  ✅ Successfully generated {total_clusters} summaries for Level {level}.{' ' * 10}")
            
        parent_embeddings = embedder.embed_chunks(next_level_chunks)
        
        all_chunks.extend(next_level_chunks)
        all_embeddings.append(parent_embeddings)
        
        current_level_chunks = next_level_chunks
        current_embeddings = parent_embeddings
        level += 1

    print(f"\n✅ RAPTOR tree completed! Total nodes (leaves + summaries): {len(all_chunks)}")

    # Global Indexing
    final_embeddings = np.vstack(all_embeddings)
    
    index_path = Path("data/raptor_summary_index")
    index_path.mkdir(parents=True, exist_ok=True)
    
    retriever.add_chunks(all_chunks, final_embeddings)
    retriever.save_index(str(index_path))
    
    print("✅ RAPTOR Index built and saved")

if __name__ == "__main__":
    build_index()