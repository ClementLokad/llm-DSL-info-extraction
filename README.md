# 🤖 LOKAD Code Analysis AI Assistant

AI assistant for LOKAD's Supply Chain Scientists to navigate Envision DSL codebases.

**Client**: [LOKAD](https://www.lokad.com) | **Framework**: École Polytechnique PSC (X24)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- API key from: [OpenAI](https://platform.openai.com), [Mistral](https://mistral.ai), or [Google AI](https://makersuite.google.com/app/apikey)

### Setup

```bash
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction

python -m venv env
.\env\Scripts\activate    # Windows
pip install -r requirements.txt

cp .env.example .env      # Add your API keys
python test.py            # Start testing
```

## 🧪 Available Models

- **GPT**: `gpt-4o-mini` (requires `OPENAI_API_KEY`)
- **Mistral**: `mistral-large-latest` (requires `MISTRAL_API_KEY`) 
- **Gemini**: `gemini-2.5-flash` (requires `GOOGLE_API_KEY`)

**Test Gemini models**: Use `python test_gemini_models.py` to find working models

Change model in `test.py`:
```python
MODEL_NAME = 'gemini'  # Options: 'gpt', 'mistral', 'gemini'
```

## 💻 Usage Examples

```python
from agents.gemini_agent import GeminiAgent

agent = GeminiAgent()
agent.initialize()

# Simple question
response = agent.generate_response("What is Python?")

# With context
response = agent.generate_response("What makes it popular?", 
                                  "Python is a programming language.")

# List available Gemini models
GeminiAgent.list_available_models()

# Different models
agent = GeminiAgent(model="gemini-2.5-flash")   # Fast, stable
agent = GeminiAgent(model="gemini-2.5-pro")    # More powerful
```

## 📊 Project Status

### ✅ Current: LLM Testing Setup
- Modular agent architecture (GPT, Mistral, Gemini)
- Interactive testing interface
- Rate limiting and error handling

### 🔄 Next: Preprocessing Pipeline
- Envision DSL parsing and chunking
- Vector embeddings for semantic search
- Metadata extraction

### 🚀 Future: RAG System
- Smart code retrieval
- Context-aware responses
- Production interface

## 🏗️ Structure

```
agents/                 # LLM implementations
├── base.py            # Abstract interface
├── gpt_agent.py       # OpenAI GPT
├── mistral_agent.py   # Mistral AI
└── gemini_agent.py    # Google Gemini

env_scripts/           # Envision DSL files (60 files)
test.py               # Testing interface
main.py               # CLI entry point
```

## 🛠️ Development

**Adding new LLM**: Inherit from `LLMAgent`, implement `initialize()`, `generate_response()`, `model_name`

**Architecture**: Modular, provider-agnostic, test-driven, scalable