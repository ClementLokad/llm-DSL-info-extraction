#!/usr/bin/env python3
"""Build search index"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.summarizers.chunk_summarizer import ChunkSummarizer
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever


def build_index():
    """
    Build search index from parsed blocks with configurable modes.
    
    Modes:
    1. Default mode: Parses blocks, generates summaries reusing already computed summaries, embeds them, and builds FAISS index 
    2. --rebuild mode: Clears existing summaries and regenerates them from scratch before building the index
    3. --check-status mode: Only checks and reports the current summary generation status
    """

    # Mode selection (mutually exclusive)
    parser = argparse.ArgumentParser(description="Build search index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Resume from previous summary file (default: False)"
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check summary generation status instead of building index"
    )
    args = parser.parse_args()

    print("🔨 Building summary index...")
    cfg = ConfigManager()
    summarizer = ChunkSummarizer(cfg.get_summarizer_config())
    parser = EnvisionParser(cfg.get_parser_config())
    chunker = EnvisionChunker(cfg.get_chunker_config())
    embedder = SentenceTransformerEmbedder(cfg.get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(cfg.get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    dirs = cfg.get('paths.input_dirs', ["env_scripts"])
    blocks = []
    chunks = []
    
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.glob("*.nvn"):
            current_blocks = parser.parse_file(str(f))
            blocks.extend(current_blocks)
            current_chunks = chunker.chunk_blocks(current_blocks, start_id=len(chunks))
            chunks.extend(current_chunks)
    
    print(f"Parsed {len(blocks)} blocks and created {len(chunks)} chunks")

    # Mode: Check status only
    if args.check_status: 
        print(summarizer.get_summary_state(chunks))

    # Mode: Build index (with optional resume)
    else:
        summarizer.generate_summary_file(chunks, rebuild=args.rebuild)
        summaries = summarizer.get_summary_list()

        embeddings = embedder.embed_batch(summaries)
        print(f"Generated {embeddings.shape[0]} embeddings")
        
        retriever.add_chunks(chunks, embeddings)
        
        index_path = Path("data/faiss_summary_index")
        index_path.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(index_path))
        
        print("✅ Summary Index built and saved")

if __name__ == "__main__":
    build_index()