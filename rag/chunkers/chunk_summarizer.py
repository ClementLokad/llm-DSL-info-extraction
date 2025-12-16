from typing import List, Dict, Any, Optional
from config_manager import get_config
from rag.core.base_chunker import CodeChunk
import agents.prepare_agent as prepare_agent

class ChunkSummarizer():
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.summary_agent = prepare_agent.prepare_chunk_summary_agent()
        self.summary_prompt = get_config().get_chunker_config().get('summary_prompt', "Summarize what the following code chunk does very briefly using keywords:")

    def summarize_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """For each chunk, adds a summary made by a LLM agent as a chunk metadata"""
        for chunk in chunks:
            chunk_summary = self.summary_agent.generate_response(self.summary_prompt + "Chunk to summarize :" + chunk.content)
            chunk.metadata['summary'] = chunk_summary