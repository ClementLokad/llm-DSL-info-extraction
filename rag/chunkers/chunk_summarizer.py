from typing import List, Dict, Any, Optional
from config_manager import get_config
from rag.core.base_chunker import CodeChunk
import agents.prepare_agent as prepare_agent

class ChunkSummarizer():
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.summary_agent = prepare_agent.prepare_chunk_summary_agent()
        self.summary_prompt = get_config().get_chunker_config().get('summary_prompt', "Summarize what the following code chunk does very briefly using keywords:")

    def summarize_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Adds a summary to the content of each chunk of the list using an LLM agent."""
        summarized_chunks = []
        for chunk in chunks:
            summary = self.summary_agent.generate_response(self.summary_prompt + "Chunk to summarize :" + chunk.content)
            summarized_chunk = f"Summary: {summary}\n\n{chunk.content}"
            summarized_chunks.append(CodeChunk(
                    content=summarized_chunk, # prepend summary to original content
                    chunk_type=chunk.chunk_type, # retain original type
                    metadata=chunk.metadata, # retain original metadata
                    context=chunk.context # retain original context
                ))
        return summarized_chunks