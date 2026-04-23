#!/usr/bin/env python3
"""Build search index"""
import os
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(os.getcwd(), "data/fastembed_models")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.config_manager import ConfigManager
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.envision_chunker import EnvisionChunker
from rag.utils.switch_db import get_default_embedder, get_default_retriever


def build_index():
    print("🔨 Building index...")
    cfg = ConfigManager()
    
    def_ret = cfg.get("retriever.type", "qdrant")
    def_emb = cfg.get("embedder.default_type", "qdrant")
    
    parser = EnvisionParser(cfg.get_parser_config())
    chunker = EnvisionChunker(cfg.get_chunker_config())
    embedder = get_default_embedder()
    embedder.initialize()
    retriever = get_default_retriever()
    if def_ret == "qdrant":
        retriever.clear_index(summary=False)
        print("Old index succesfully emptied")
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

    if def_ret == def_emb == "qdrant":
        embeddings = embedder.embed_chunks_hybrid(chunks)
    else:
        embeddings = embedder.embed_chunks(chunks)

    print(f"Generated {len(embeddings)} embeddings")
    
    if def_ret == def_emb == "qdrant":
        retriever.add_chunks_hybrid(chunks, embeddings, summary=False)
    elif def_ret == "qdrant":
        retriever.add_chunks(chunks, embeddings, summary=False)
    else:
        retriever.add_chunks(chunks, embeddings)

    if def_ret == "qdrant":
        retriever.close()
    else:
        index_path = Path("data/faiss_index")
        index_path.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(index_path))

    print("✅ Index built and saved")


if __name__ == "__main__":
    build_index()
