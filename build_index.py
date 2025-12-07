#!/usr/bin/env python3
"""Build search index"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever


def build_index():
    print("🔨 Building index...")
    cfg = ConfigManager()
    
    parser = EnvisionParser(cfg.get_parser_config())
    chunker = SemanticChunker(cfg.get_chunker_config())
    embedder = SentenceTransformerEmbedder(cfg.get_embedder_config())
    embedder.initialize()
    retriever = FAISSRetriever(cfg.get_retriever_config())
    retriever.initialize(embedder.embedding_dimension)
    
    dirs = cfg.get('paths.input_dirs', ["env_scripts"])
    blocks = []
    
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.glob("*.nvn"):
            blocks.extend(parser.parse_file(str(f)))
    
    print(f"Parsed {len(blocks)} blocks")
    
    chunks = chunker.chunk_blocks(blocks)
    print(f"Created {len(chunks)} chunks")

    embeddings = embedder.embed_chunks(chunks)
    print(f"Generated {embeddings.shape[0]} embeddings")
    
    retriever.add_chunks(chunks, embeddings)
    
    index_path = Path("data/faiss_index")
    index_path.mkdir(parents=True, exist_ok=True)
    retriever.save_index(str(index_path))
    
    print("✅ Index built and saved")


if __name__ == "__main__":
    build_index()
