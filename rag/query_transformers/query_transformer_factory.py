from rag.query_transformers.fusion_query_transformer import FusionQueryTransformer
from rag.query_transformers.hyde_query_transformer import HydeQueryTransformer

# Mapping linking flags to actual query transform classes
TRANSFORMER_MAPPING = {
    "hyde": HydeQueryTransformer,
    "fusion": FusionQueryTransformer
}
class QueryTransformerFactory:
    @staticmethod
    def create(config: dict):
        transformer_type = config.get('query_transformer.query_transformer_mode', None)
        
        if not transformer_type:
            return
        
        transformer_class = TRANSFORMER_MAPPING.get(transformer_type)
        
        if not transformer_class:
            raise ValueError(f"Transformer type '{transformer_type}' not recognized")
            
        return transformer_class(config)