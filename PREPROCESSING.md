# Envision DSL Preprocessing Pipeline

This preprocessing module provides a complete pipeline for analyzing LOKAD's Envision DSL codebases using natural language queries. It includes parsing, semantic chunking, embedding generation, and similarity-based retrieval.

## Architecture Overview

The preprocessing pipeline consists of four main components:

1. **Parser**: Analyzes Envision DSL files (.nvn) and extracts semantic code blocks
2. **Chunker**: Groups related code blocks into semantically meaningful chunks
3. **Embedder**: Converts code chunks into vector embeddings for similarity search
4. **Retriever**: Stores embeddings and performs fast similarity search

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from preprocessing_pipeline import PreprocessingPipeline

# Create pipeline with default configuration
pipeline = PreprocessingPipeline()

# Initialize all components
pipeline.initialize_components()

# Process Envision files
pipeline.process_files(['path/to/your/file.nvn'])

# Search for relevant code
results = pipeline.search("How to calculate demand forecasting?", top_k=5)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Type: {result.chunk.chunk_type}")
    print(f"Content: {result.chunk.content[:200]}...")
    print()
```

### 3. Command Line Usage

```bash
# Process files and build index
python preprocessing_pipeline.py process --input env_scripts/ --output ./index

# Search the index
python preprocessing_pipeline.py search --query "demand forecasting" --top-k 5

# Run a quick test
python preprocessing_pipeline.py test
```

## Components

### Parser (EnvisionParser)

Analyzes Envision DSL files and extracts different types of code blocks:

- **Read statements**: Data ingestion operations
- **Table definitions**: Table operations and transformations
- **Assignments**: Variable calculations and assignments
- **Show statements**: Visualizations and outputs
- **Comment blocks**: Documentation sections

```python
from preprocessing.parsers.envision_parser import EnvisionParser

parser = EnvisionParser()
blocks = parser.parse_file('script.nvn')

for block in blocks:
    print(f"{block.block_type}: {block.name} (lines {block.line_start}-{block.line_end})")
```

### Chunker (SemanticChunker)

Groups related code blocks into chunks that respect semantic boundaries:

- Groups by section (major workflow sections)
- Keeps related assignments together
- Separates read statements for clarity
- Includes contextual comments
- Respects token limits for embedding

```python
from preprocessing.chunkers.semantic_chunker import SemanticChunker

chunker = SemanticChunker({
    'max_chunk_tokens': 512,
    'group_by_section': True,
    'include_context_comments': True
})

chunks = chunker.chunk_blocks(blocks)
```

### Embedders

Three embedder options with different trade-offs:

#### SentenceTransformerEmbedder (Recommended)
- **Pros**: Fast, no API costs, good performance
- **Cons**: Larger memory usage, local processing only
- **Best for**: High-volume processing, development

```python
from preprocessing.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder({
    'model_name': 'all-MiniLM-L6-v2',  # Fast and efficient
    'batch_size': 32
})
embedder.initialize()
```

#### OpenAIEmbedder
- **Pros**: Highest quality embeddings
- **Cons**: API costs, rate limits, requires internet
- **Best for**: Production use when quality is critical

```python
from preprocessing.embedders.openai_embedder import OpenAIEmbedder

embedder = OpenAIEmbedder({
    'model_name': 'text-embedding-3-small',
    'api_key': 'your-openai-key'  # Or set OPENAI_API_KEY env var
})
embedder.initialize()
```

#### GeminiEmbedder
- **Pros**: Good quality, Google ecosystem integration
- **Cons**: API costs, rate limits, quota restrictions
- **Best for**: Alternative to OpenAI with different pricing

```python
from preprocessing.embedders.gemini_embedder import GeminiEmbedder

embedder = GeminiEmbedder({
    'model_name': 'models/text-embedding-004',
    'api_key': 'your-google-key'  # Or set GOOGLE_API_KEY env var
})
embedder.initialize()
```

### Retriever (FAISSRetriever)

Provides fast similarity search using Facebook's FAISS library:

```python
from preprocessing.retrievers.faiss_retriever import FAISSRetriever

retriever = FAISSRetriever({
    'index_type': 'IndexFlatIP',  # Exact search for best quality
    'top_k': 10
})

retriever.initialize(embedder.embedding_dimension)
retriever.add_chunks(chunks, embeddings)

# Search with natural language
results = retriever.search_by_text("demand calculation", embedder, top_k=5)
```

## Configuration

Create a YAML configuration file to customize the pipeline:

```yaml
# config.yaml
embedder:
  type: sentence_transformer
  model_name: all-MiniLM-L6-v2
  batch_size: 32

chunker:
  max_chunk_tokens: 512
  group_by_section: true

retriever:
  index_type: IndexFlatIP
  top_k: 10
```

Use with the pipeline:

```python
from preprocessing.utils.config import ConfigManager

config = ConfigManager('config.yaml')
pipeline = PreprocessingPipeline(config)
```

## Testing

Run individual component tests:

```bash
# Test the parser
python preprocessing/tests/test_parser.py

# Test the chunker
python preprocessing/tests/test_chunker.py

# Test embeddings (requires sentence-transformers)
python preprocessing/tests/test_embedder.py
```

## Performance Considerations

### Memory Usage
- **Sentence-transformers**: ~500MB-2GB depending on model
- **Embeddings**: ~4 bytes × dimensions × chunks (e.g., 384 dims × 1000 chunks = ~1.5MB)
- **FAISS index**: Similar to embeddings, plus small overhead

### Speed
- **Parsing**: ~1000 lines/second
- **Chunking**: ~500 chunks/second
- **Embedding (local)**: ~100-1000 chunks/second depending on model
- **Embedding (API)**: Limited by rate limits (varies by provider)
- **Search**: <1ms for exact search, microseconds for approximate

### Optimization Tips

1. **Use sentence-transformers for development** to avoid API costs
2. **Choose appropriate chunk sizes** (256-512 tokens work well)
3. **Use IndexIVFFlat for large datasets** (>10K chunks)
4. **Enable GPU acceleration** for FAISS if available
5. **Batch process files** to improve efficiency

## Troubleshooting

### Common Issues

**"sentence-transformers not found"**
```bash
pip install sentence-transformers
```

**"faiss-cpu not found"**
```bash
pip install faiss-cpu
```

**"OpenAI API key not found"**
```bash
export OPENAI_API_KEY="your-key-here"
```

**"Out of memory during embedding"**
- Reduce batch_size in embedder config
- Use a smaller model (e.g., all-MiniLM-L6-v2)
- Process files in smaller batches

**"Slow search performance"**
- Use IndexIVFFlat instead of IndexFlatIP for large datasets
- Consider GPU acceleration with faiss-gpu
- Reduce embedding dimensions if possible

### Debug Mode

Enable verbose logging for troubleshooting:

```python
from preprocessing.utils.helpers import setup_logging
setup_logging(level='DEBUG')
```

## API Reference

For detailed API documentation, see the docstrings in each module:

- `preprocessing.core.*`: Abstract base classes
- `preprocessing.parsers.*`: Parser implementations
- `preprocessing.chunkers.*`: Chunker implementations  
- `preprocessing.embedders.*`: Embedder implementations
- `preprocessing.retrievers.*`: Retriever implementations
- `preprocessing.utils.*`: Utility functions and configuration

## Contributing

When adding new components:

1. Inherit from the appropriate base class in `preprocessing.core`
2. Implement all abstract methods
3. Add comprehensive docstrings
4. Create test scripts in `preprocessing.tests`
5. Update this README with usage examples

## License

This preprocessing module is part of the LLM DSL Info Extraction project.