from typing import List, Dict, Any, Optional
import agents.prepare_agent as prepare_agent
from rag.core.base_query_transformer import BaseQueryTransformer
import time

class FusionQueryTransformer(BaseQueryTransformer):
    """
    Query transformer taking a question and transforming it into several sub queries in order to improve retrieval
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.agent = prepare_agent.prepare_query_transformer_agent()
        self.rate_limit_delay = config.get('agent.rate_limit_delay', 0)
        self.fusion_prompt = f"Take the following complex question and decompose it into {self.generated_instances_amount} distinct sub-questions. Your response must only be the juxtaposition of these sub-questions, with each one separated by a $ character. Do not add any preamble, explanation, or other text \n"

    def transform(self, query : str) -> list[str]:
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        sub_queries = self.agent.generate_response(self.fusion_prompt + query)
        sub_queries_list = sub_queries.split("$")
        return sub_queries_list