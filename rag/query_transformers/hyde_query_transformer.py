from typing import List, Dict, Any, Optional
import agents.prepare_agent as prepare_agent
from rag.core.base_query_transformer import BaseQueryTransformer
import time

class HydeQueryTransformer(BaseQueryTransformer):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.index_type = config.get('embedder.index_type', "full_chunk")

    def transform(self, query : str) -> list[str]:
        if self.index_type == "summaries":
            self.hyde_prompt = f"Act as an Envision DSL expert. Generate {self.generated_instances_amount} concise technical summaries (max 150 words) of a script answering this query: {query}. Your response must only be the juxtaposition of these summaries, with each one separated by a $ character. Do not add any preamble, explanation, or other text \n"

        #IN CASE WE WANT TO ADAPT THE PROMPT TO INDEX TYPE
        # elif self.index_type == "full_chunk":
        #     self.hyde_prompt = f"Take the following complex question and decompose it into {self.generated_instances_amount} distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text \n"

        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        sub_queries = self.agent.generate_response(self.hyde_prompt + query)
        sub_queries_list = sub_queries.split("$")
        return sub_queries_list