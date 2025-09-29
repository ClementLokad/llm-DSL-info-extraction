# 🤖 LOKAD Code Analysis AI Assistant

## 🎯 Project Objective

This project aims to create an advanced system to assist **LOKAD**'s *Supply Chain Scientists*. The goal is to help them navigate, understand, and query a complex codebase written in **Envision**, LOKAD's proprietary language, using natural language questions.

### 🏗️ Project Structure

```text
llm-DSL-info-extraction/
├── 📁 preprocessing/           # Preprocessing pipeline (COMPLETED ✅)
│   ├── pipeline.py            # Complete preprocessing with all embedders
│   └── __init__.py
├── 📁 agents/                 # AI agents for query phase (FUTURE)
│   ├── base.py               # LLMAgent abstract interface
│   ├── gpt_agent.py          # OpenAI implementation
│   ├── mistral_agent.py      # Mistral implementation
│   └── gemini_agent.py       # Google Gemini implementation
├── 📁 env_scripts/           # Envision DSL files (60 files)
├── 📁 processed_data/        # Preprocessing results
├── 📄 process.py             # Main preprocessing CLI
├── 📄 analyze.py             # Results analysis tool
├── 📄 test.py               # Agent testing script
├── 📄 requirements.txt       # Dependencies
├── 📄 .env.example          # Configuration template
└── 📄 README.md             # This documentation
```

* **Client**: [LOKAD](https://www.lokad.com)
* **Framework**: Collective Scientific Project (PSC) - École Polytechnique (X24)

## � Quick Start - Preprocessing Phase

The preprocessing phase is **complete and ready to use**! It processes Envision DSL files and creates vector embeddings for semantic search.

### Basic Usage

```bash
# Process files with mock embedder (for testing - no API key needed)
python process.py --model mock --max-files 5

# Process files with real embedders (API key required)
python process.py --model gemini    # Uses Google Gemini embeddings
python process.py --model openai    # Uses OpenAI embeddings

# Analyze processed results
python analyze.py --file processed_data/results_mock.pkl
```

### Available Models (Auto-Discovered)

The system automatically discovers all embedding models in the `agents/` directory:

* **`mock`**: No API key required, generates consistent test embeddings (768-dim)
* **`gemini`**: Google Gemini embeddings (768-dim) - requires `GOOGLE_API_KEY`
* **`openai`**: OpenAI embeddings (1536-dim) - requires `OPENAI_API_KEY`
* **`huggingface`**: Sentence Transformers embeddings (384-dim) - requires `sentence-transformers` package

**Adding New Models**: Simply create a new `*_embedder.py` file in `agents/` following the `BaseEmbedder` interface!

### Command Line Options

```bash
python process.py --help

Options:
  --script-dir     Directory with Envision scripts (default: env_scripts)
  --output-dir     Output directory (default: processed_data)  
  --model          Model type: mock, gemini, openai (default: mock)
  --chunk-size     Chunk size in characters (default: 512)
  --overlap        Chunk overlap in characters (default: 50)
  --max-files      Max files to process for testing (optional)
```

## �💻 Prerequisites

Before starting, make sure you have:

* Python 3.10 or higher installed
* pip (Python package manager)
* Git
* OpenAI API key (to use GPT-4)
* Mistral AI API key (to use Mistral)
* Google AI Studio API key (to use Gemini)

### Getting API Keys

1. **OpenAI (for GPT)**
   * Create an account on [OpenAI Platform](https://platform.openai.com)
   * Go to the API Keys section to create your key

2. **Mistral AI**
   * Sign up on [Mistral AI](https://mistral.ai)
   * Generate your API key in account settings

3. **Google AI Studio (for Gemini)**
   * Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   * Create a project if needed
   * Generate an API key in the "API & Services" section

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction
```

### 2. Python Environment Setup

Create and activate the virtual environment:

**Windows:**
```bash
python -m venv env
.\env\Scripts\activate
```

**Unix/macOS:**
```bash
python -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. API Keys Configuration (Optional)

For testing, you can use the `mock` embedder (no API key needed). For production with real embeddings:

1. Copy the configuration template:

```bash
# Windows
copy .env.example .env

# Unix/macOS  
cp .env.example .env
```

2. Add your API keys to `.env`:
   * Create an account on [OpenAI Platform](https://platform.openai.com)
   * Sign up on [Mistral AI](https://mistral.ai)
   * Create an account on [Google AI Studio](https://makersuite.google.com/app/apikey)
   * Edit the `.env` file with your keys:

```env
OPENAI_API_KEY=your-openai-api-key
MISTRAL_API_KEY=your-mistral-api-key
GOOGLE_API_KEY=your-google-api-key
```

### 5. Testing the Models

A test script is provided to verify the proper functioning of the different models:

```bash
python test.py
```

To choose which model to test, modify the `MODEL_NAME` variable in `test.py`:
```python
# Possible values: 'gpt', 'mistral', 'gemini'
MODEL_NAME = 'gpt'  # Change this value according to the model you want to test
```

Characteristics of different models:

1. **GPT (OpenAI)**
   * Most mature model
   * Excellent context understanding
   * Advanced multilingual support

2. **Mistral**
   * Open source alternative
   * Good overall performance
   * Lightweight and efficient model

3. **Gemini (Google)**
   * Latest generation from Google
   * Excellent code understanding
   * Adjustable generation parameters

If no errors appear during the test, your environment is ready!

### Usage Examples

```python
# Example with GPT-4
from agents import GPTAgent

agent = GPTAgent("your-openai-api-key")
response = agent.process_question("What does the `calculate_stock_level` function do?")
print(response)

# Example with Mistral
from agents import MistralAgent

agent = MistralAgent("your-mistral-api-key")
response = agent.process_question("Explain the code of the `optimize_forecast` function")
print(response)

# Example with Gemini
from agents import GeminiAgent

agent = GeminiAgent("your-google-api-key")
# Temperature customization (0 = more precise, 1 = more creative)
agent.set_temperature(0.7)
response = agent.process_question("Analyze the `validate_input_data` function")
print(response)

# Metadata extraction with Gemini
metadata = agent.extract_metadata("def calculate_total(items: List[Item]) -> float:")
print(metadata)  # Display structured code metadata
```

### Advanced Configuration

#### Gemini Generation Parameters

The Gemini agent offers configurable parameters to adjust text generation:

```python
from agents import GeminiAgent

agent = GeminiAgent("your-google-api-key")

# Temperature adjustment (creativity)
agent.set_temperature(0.3)  # More conservative
agent.set_temperature(0.9)  # More creative

# Default parameters are:
# temperature = 0.7
# top_p = 0.95
# top_k = 40
# max_output_tokens = 2048
```

### Detailed File Structure

```
llm-DSL-info-extraction/
├── agents/                 # AI agents package
│   ├── __init__.py
│   ├── base.py            # LLMAgent abstract interface
│   ├── gpt_agent.py       # OpenAI implementation
│   └── mistral_agent.py   # Mistral implementation
├── .env.example           # Configuration template
├── .gitignore            # Files ignored by git
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

### System Requirements

Before starting, make sure you have:

* Python 3.10 or higher installed
* pip (Python package manager)
* Git
* OpenAI API key (to use GPT-4) (at least one API key)
* Mistral AI API key (to use Mistral) (at least one API key)

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction
```

2. **Create a Python virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate   # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

```bash
# Copy example file
cp .env.example .env

# Edit .env with your API keys
# You will need:
# - OpenAI API key (https://platform.openai.com)
# - Mistral AI API key (https://mistral.ai)
```

## 🔄 System Architecture

### Phase 1: 📚 Preprocessing Pipeline (✅ COMPLETED)

The preprocessing phase transforms raw Envision scripts into a searchable knowledge base:

```
Envision Files (.nvn) → Chunking → Embeddings → Vector Database
     📄 60+ files    →   📝 ~30k chunks  →  🧠 Vector Search  →  📊 Results (.pkl)
```

**Components:**
* **EnvisionProcessor**: Parses and cleans Envision DSL syntax
* **EnvisionChunker**: Intelligently splits code into semantic chunks (512 chars, 50 overlap)
* **Multiple Embedders**: Gemini, OpenAI, or Mock embedder for vector generation
* **VectorSearch**: FAISS-based similarity search with metadata
* **Results Storage**: Serialized results with embeddings and search index

**Features:**
* 🔄 **Model-agnostic design**: Switch between embedding providers easily
* 🚀 **Production ready**: Handles large codebases efficiently  
* 🧪 **Testing support**: Mock embedder for development without API costs
* 📊 **Rich analysis**: Pattern detection and content statistics

### Phase 2: 💡 Query and Response System (FUTURE)

This phase will handle user questions and generate intelligent responses:

```
Natural Language Question → RAG Retrieval → LLM Response → Validation → Final Answer
```

**Planned Components:**
* **Question Processing**: Parse and understand user intent
* **RAG System**: Retrieve most relevant code chunks using vector similarity
* **Multi-LLM Support**: GPT-4, Mistral, Gemini for response generation
* **Logic Checker**: Validate response accuracy and code syntax
* **Answer Grader**: Quality assessment and benchmarking

## 📊 Preprocessing Results

Current preprocessing capabilities (tested with 60 Envision files):

* **Files processed**: 60 Envision DSL files
* **Total chunks generated**: ~29,565 semantic chunks
* **Average chunk size**: ~58 characters (with 512 max)
* **Pattern detection**: Functions, variables, calculations, tables, exports
* **Processing time**: ~2-3 minutes for full codebase
* **Output format**: Pickle files with embeddings, metadata, and search index

### Sample Analysis Output

```
PROCESSING STATISTICS
Files processed: 3
Total chunks: 1532
Embedder used: MockEmbedder
Average chunk size: 58.0 characters

ENVISION CODE PATTERNS
- Read Statements: 7 chunks (0.5%)
- Const Declarations: 12 chunks (0.8%) 
- Export Statements: 2 chunks (0.1%)
- Table Definitions: 43 chunks (2.8%)
- Calculations: 467 chunks (30.5%)
```

## 🎯 Current Status & Next Steps

### ✅ Completed (Phase 1 - Preprocessing)

* **Complete preprocessing pipeline** with model-agnostic design
* **Three embedding providers**: Gemini, OpenAI, Mock (for testing)
* **Intelligent code chunking** for Envision DSL syntax
* **Vector search capabilities** with FAISS integration
* **Comprehensive analysis tools** for processed data
* **Production-ready codebase** with clean, consolidated architecture

### 🔄 In Development (Phase 2 - Query System)

* **Multi-LLM query agents** (GPT-4, Mistral, Gemini) - foundation ready
* **RAG system integration** to connect preprocessing with query handling
* **Response validation** and quality assessment
* **Interactive query interface** for Supply Chain Scientists

### 🚀 Getting Started

1. **For testing/development**: Start with `python process.py --model mock --max-files 5`
2. **For production**: Set up API keys and run `python process.py --model gemini`
3. **Analyze results**: Use `python analyze.py --file processed_data/results_*.pkl`

The preprocessing phase is **complete and production-ready**! 🎉

## 🔧 Adding New Embedding Models

The system uses a **plugin architecture** - adding new embedding models is extremely simple:

### Step 1: Create a new embedder file

Create `agents/your_model_embedder.py`:

```python
from typing import List
from .base import BaseEmbedder

class YourModelEmbedder(BaseEmbedder):
    def __init__(self, api_key: str = "", **kwargs):
        # Initialize your model here
        self._dimension = 512  # Your model's dimension
    
    def embed(self, text: str) -> List[float]:
        # Implement single text embedding
        return [0.0] * self._dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Implement batch embedding
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "yourmodel"  # This becomes the CLI argument
```

### Step 2: Use immediately

```bash
# Your new model is automatically discovered!
python process.py --model yourmodel --api-key your-api-key

# Or set environment variable
export YOURMODEL_API_KEY=your-key
python process.py --model yourmodel
```

**That's it!** No configuration files, no registry updates, no core code changes needed.
