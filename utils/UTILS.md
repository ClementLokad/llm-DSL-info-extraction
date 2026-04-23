# 🔧 Utils - Configuration & File Mapping

> Utility modules for configuration management and file path resolution across the project.

---

## 📁 Contents

### **`config_manager.py`** - Centralized Configuration

**Purpose**: Load and manage configuration from YAML + environment variables.

**Key Features**:
- Load from `config.yaml` (main settings)
- Load from `.env` (secrets, API keys)
- Precedence: env vars > YAML > defaults
- Singleton pattern: `get_config()` returns cached instance

**API**:
```python
from utils.config_manager import get_config

config = get_config()

# Access by path (dot notation)
max_tokens = config.get('chunker.max_chunk_tokens')
api_key = config.get('agents.claude.api_key')

# Get config sections
chunker_cfg = config.get_chunker_config()
embedder_cfg = config.get_embedder_config()
retriever_cfg = config.get_retriever_config()

# Get API keys by model
api_key = config.get_api_key('claude')
api_key = config.get_api_key('mistral')

# Get benchmark type
benchmark_type = config.get_benchmark_type()
```

**Configuration Structure**:
```yaml
chunker:
  max_chunk_tokens: 512
  overlap_lines: 3

embedder:
  type: "qdrant"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  
retrieval:
  top_k: 5
  
agents:
  claude:
    api_key: ${CLAUDE_API_KEY}  # From .env
  mistral:
    api_key: ${MISTRAL_API_KEY}
```

---

### **`get_mapping.py`** - File Path Resolution

**Purpose**: Map file IDs to original file paths and build directory trees.

**Key Functions**:

```python
def get_file_mapping(mapping_file_path=None) -> Dict[str, str]:
    """
    Load mapping: file_id → original_path
    
    Format: CSV with columns [file_id, original_path]
    Example:
      48623,/users/dev/items.nvn
      48624,/users/dev/orders.nvn
    
    Returns: {
        "48623": "/users/dev/items.nvn",
        "48624": "/users/dev/orders.nvn"
    }
    """

def get_inverse_mapping(mapping_file_path=None) -> Dict[str, str]:
    """
    Invert mapping: original_path → file_id
    
    Returns: {
        "/users/dev/items.nvn": "48623",
        "/users/dev/orders.nvn": "48624"
    }
    """

def build_file_tree(mapping_file_path=None) -> Dict[str, str]:
    """
    Build nested directory tree from paths.
    
    Returns hierarchical dict:
    {
        "env_scripts": {
            "items.nvn": "...",
            "orders.nvn": "...",
            "subfolder": {...}
        }
    }
    """
```

**Usage**:
```python
from utils.get_mapping import get_file_mapping, build_file_tree

# Get ID→path mapping
mapping = get_file_mapping()
path = mapping["48623"]  # "/users/dev/items.nvn"

# Build tree for display
tree = build_file_tree()
# Used by FileTreeTool in agent_workflow
```

**Why Mapping?**
- Codebase uses hashed file IDs internally (performance)
- Mapping file translates IDs back to human-readable paths
- Critical for displaying results to users (show real paths, not IDs)

---

## 🚀 Usage Pattern

```python
# In any module
from utils.config_manager import get_config
from utils.get_mapping import get_file_mapping, build_file_tree

# Configuration
config = get_config()
top_k = config.get('rag.top_k_chunks')

# File mapping
file_tree = build_file_tree()
mapping = get_file_mapping()

# In results
for result in results:
    file_id = result['file_id']
    path = mapping[file_id]  # Convert to readable path
```

---

**Lightweight utilities for configuration and file resolution** ✅
