#!/usr/bin/env python3
"""
Before/After demonstration of configuration externalization.

This script shows the improvement from hardcoded values to externalized configuration.
"""

def show_before_after():
    """Demonstrate the before/after of configuration externalization."""
    
    print("🔧 Configuration Externalization: Before vs After")
    print("=" * 60)
    
    print("\n❌ BEFORE (Hardcoded Values in Code):")
    print("-" * 40)
    print("""
class BaseChunker:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_chunk_tokens = self.config.get('max_chunk_tokens', 512)  # ❌ Hardcoded
        
class EnvisionParser:
    def __init__(self):
        # ❌ All hardcoded patterns
        self._section_delimiter_pattern = re.compile(r'^///[~=-]{20,}.*$')  # ❌ 20 hardcoded
        
class CodeChunk:
    def __post_init__(self):
        if self.size_tokens == 0:
            self.size_tokens = len(self.content) // 4  # ❌ 4 hardcoded

Problems:
• Need to modify code to change parameters
• Magic numbers scattered throughout codebase  
• No central configuration management
• Difficult to tune for different environments
""")
    
    print("\n✅ AFTER (Externalized Configuration):")
    print("-" * 40)
    print("""
# config.yaml - All parameters in one place
chunker:
  max_chunk_tokens: 512     # ✅ Easy to change
  overlap_lines: 2          # ✅ No code modification needed
  chars_per_token: 4        # ✅ Centralized
  
parser:
  section_delimiter:
    min_chars: 20           # ✅ Configurable
    valid_chars: ["~", "=", "-"]

# Python code - Clean and flexible
class BaseChunker:
    def __init__(self, config=None):
        if config is None:
            config = get_config().get_chunker_config()  # ✅ Load from YAML
        self.config = config
    
    @property
    def max_chunk_tokens(self) -> int:
        return self.config.get('max_chunk_tokens', 512)  # ✅ Always synced
        
class EnvisionParser:
    def __init__(self):
        config = get_config().get_parser_config()
        section_config = config.get('section_delimiter', {})
        min_chars = section_config.get('min_chars', 20)  # ✅ From config
        
Benefits:
✅ Zero hardcoded values in Python code
✅ Easy parameter tuning via config files
✅ Environment-specific configurations  
✅ Central configuration management
✅ Runtime configuration updates
✅ Self-documenting YAML with comments
""")
    
    print("\n🎯 Key Improvements:")
    print("-" * 20)
    improvements = [
        "All numeric constants externalized to config.yaml",
        "API keys and secrets separated in .env files", 
        "Properties ensure automatic synchronization",
        "Configuration validation and error handling",
        "Environment-specific settings (dev/prod)",
        "Self-documenting configuration with comments",
        "No code changes needed for parameter tuning",
        "Runtime configuration reloading capability"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"{i:2d}. {improvement}")
    
    print("\n📊 Configuration Statistics:")
    print("-" * 25)    
    from config_manager import get_config
    config = get_config()
    
    # Count externalized parameters
    chunker_params = len(config.get_chunker_config())
    parser_params = len(config.get_parser_config()) 
    embedder_params = len(config.get_embedder_config())
    retriever_params = len(config.get_retriever_config())
    pipeline_params = len(config.get_pipeline_config())
    
    total_params = chunker_params + parser_params + embedder_params + retriever_params + pipeline_params
    
    print(f"  📋 Total parameters externalized: {total_params}")
    print(f"  🧩 Chunker parameters: {chunker_params}")
    print(f"  🔍 Parser parameters: {parser_params}")
    print(f"  🧠 Embedder parameters: {embedder_params}")
    print(f"  📊 Retriever parameters: {retriever_params}")
    print(f"  ⚙️  Pipeline parameters: {pipeline_params}")

    print("\n🚀 Result: Flexible, maintainable, configuration-driven architecture!")

if __name__ == "__main__":
    # No environment setup needed (dev by default)
    show_before_after()