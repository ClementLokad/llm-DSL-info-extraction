import tiktoken
from config_manager import get_config


def get_token_count(content: str, encoding_name: str = "cl100k_base") -> int:
    """Returns approximate token count for this block."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(content))
    except Exception:
        print("Error: couldn't count tokens, resulting to default approximation")
        
        chars_per_token = get_config().get('embedder.text_preparation.chars_per_token_code', 3)
        return len(content)//chars_per_token
    
    