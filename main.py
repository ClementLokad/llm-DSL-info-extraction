#!/usr/bin/env python3
"""DSL Query System"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from router import Router, QueryType
from grep.searcher import GrepSearcher
from rag.parsers.envision_parser import EnvisionParser
from rag.chunkers.semantic_chunker import SemanticChunker
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrievers.faiss_retriever import FAISSRetriever


class DSLQuerySystem:
    def __init__(self):
        self.config = ConfigManager()
        self.router = None
        self.grep = None
        self.rag = {}
        self.agent = None
        
    def initialize(self, verbose=True):
        if verbose:
            print("🚀 Initializing...")
            
        agent_type = self.config.get_default_agent()
        if agent_type == 'mistral':
            from agents.mistral_agent import MistralAgent
            self.agent = MistralAgent()
        elif agent_type == 'gemini':
            from agents.gemini_agent import GeminiAgent
            self.agent = GeminiAgent()
        else:
            from agents.gpt_agent import GPTAgent
            self.agent = GPTAgent()
            
        self.agent.initialize()
        self.router = Router(self.agent)
        
        dirs = self.config.get('paths.input_dirs', ["env_scripts"])
        self.grep = GrepSearcher(dirs)
        
        parser = EnvisionParser(self.config.get_parser_config())
        chunker = SemanticChunker(self.config.get_chunker_config())
        embedder = SentenceTransformerEmbedder(self.config.get_embedder_config())
        embedder.initialize()
        retriever = FAISSRetriever(self.config.get_retriever_config())
        retriever.initialize(embedder.embedding_dimension)
        
        index_path = Path("data/faiss_index")
        metadata_file = index_path / "metadata.pkl"
        
        if not metadata_file.exists():
            if verbose:
                print("Building index...")
            blocks = []
            for d in dirs:
                p = Path(d)
                if p.exists():
                    for f in p.glob("*.nvn"):
                        blocks.extend(parser.parse_file(str(f)))
            chunks = chunker.chunk_blocks(blocks)
            embs = embedder.embed_chunks(chunks)
            retriever.add_chunks(chunks, embs)
            index_path.mkdir(parents=True, exist_ok=True)
            retriever.save_index(str(index_path))
        else:
            retriever.load_index(str(index_path))
            
        self.rag = {'embedder': embedder, 'retriever': retriever}
        if verbose:
            print("✅ Ready\n")
            
    def query(self, question, verbose=False):
        c = self.router.classify(question)
        if verbose:
            print(f"🎯 {c.qtype.value} ({c.confidence:.0%})")
            
        if c.qtype == QueryType.GREP:
            r = self.grep.search(c.pattern or "")
            return self.grep.format_answer(r, question)
        else:
            emb = self.rag['embedder'].embed_text(question)
            results = self.rag['retriever'].search(emb, top_k=5)
            ctx = "\n\n".join([f"[{r.chunk.metadata.get('file_path', 'unknown')}]\n{r.chunk.content}" for r in results])
            return self.agent.generate_response(question, ctx)
            
    def interactive(self):
        print("\n💬 Interactive (exit to quit)")
        print("=" * 60)
        while True:
            try:
                q = input("\n❓ ").strip()
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                if q:
                    print(f"\n💡 {self.query(q, verbose=True)}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ {e}")
        print("\n👋")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', '-q')
    p.add_argument('--verbose', '-v', action='store_true')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    
    sys = DSLQuerySystem()
    sys.initialize(verbose=not args.quiet)
    
    if args.query:
        print(sys.query(args.query, verbose=args.verbose))
    else:
        sys.interactive()


if __name__ == "__main__":
    main()
