#!/usr/bin/env python3
"""Build search index"""
import os
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(os.getcwd(), "data/fastembed_models")

import sys
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.config_manager import ConfigManager
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.summarizers.chunk_summarizer import ChunkSummarizer
from rag.utils.switch_db import get_default_embedder, get_default_retriever


def _load_existing_summaries(summary_list_path: str) -> dict:
    """Helper to load existing summaries from JSON without initializing ChunkSummarizer."""
    if not os.path.exists(summary_list_path) or os.path.getsize(summary_list_path) == 0:
        return {}
    try:
        with open(summary_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


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
        help="Don't resume from previous summary file (default: False)"
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check summary generation status instead of building index"
    )
    args = parser.parse_args()

    cfg = ConfigManager()

    def_ret = cfg.get("retriever.type", "qdrant")
    def_emb = cfg.get("embedder.default_type", "qdrant")
    
    print("🔨 Building summary index...")
    parser = EnvisionParser(cfg.get_parser_config())
    chunker = EnvisionChunker(cfg.get_chunker_config())
    embedder = get_default_embedder()
    embedder.initialize()
    retriever = get_default_retriever()
    if def_ret == "qdrant":
        retriever.clear_index(summary=True)
        print("Old summary index succesfully emptied")
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
        summarizer = ChunkSummarizer(cfg.get_summarizer_config())
        print(summarizer.get_summary_state(chunks))

    # Mode: Build index (with optional resume)
    else:
        if len(chunks) == 0:
            print("⚠️ No chunks found, skipping LLM initialization and summary generation")
            return
        
        # Check if all summaries are already computed and not rebuilding
        summary_config = cfg.get_summarizer_config()
        summary_list_path = summary_config.get('summary_list_path', "summaries.json")
        existing_summaries = _load_existing_summaries(summary_list_path)
        
        if len(existing_summaries) == len(chunks) and not args.rebuild:
            print(f"✅ All {len(chunks)} summaries already computed. Loading from cache...")
            summarizer = None  # Don't load the LLM
        else:
            print(f"📝 {len(existing_summaries)}/{len(chunks)} summaries already computed. Generating missing summaries...")
            summarizer = ChunkSummarizer(cfg.get_summarizer_config())
            summarizer.generate_summary_file(chunks, rebuild=args.rebuild)
        
        # Load the summaries (either from existing or newly generated)
        if summarizer is None:
            # All summaries were already cached, just load them
            summary_list = [existing_summaries[str(i)] for i in range(len(chunks))]
        else:
            # Some summaries were generated, get the updated list
            summary_list = summarizer.get_summary_list()
        
        # Assign summaries to chunk metadata
        for i, summary in enumerate(summary_list):
            chunks[i].metadata['summary'] = summary

        if def_ret == def_emb == "qdrant":
            embeddings = embedder.embed_chunks_hybrid(chunks)
        else:
            embeddings = embedder.embed_chunks(chunks)      
        print(f"Generated {len(embeddings)} embeddings from summaries if possible or raw chunks otherwise")
        
        if def_ret == def_emb == "qdrant":
            retriever.add_chunks_hybrid(chunks, embeddings, summary=True)
        elif def_ret == "qdrant":
            retriever.add_chunks(chunks, embeddings, summary=True)
        else:
            retriever.add_chunks(chunks, embeddings)

        if def_ret == "qdrant":
            retriever.close()
        else:
            index_path = Path("data/faiss_summary_index")
            index_path.mkdir(parents=True, exist_ok=True)
            retriever.save_index(str(index_path))
        
        print("✅ Summary Index built and saved")

if __name__ == "__main__":
    build_index()