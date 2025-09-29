# 🤖 LOKAD Code Analysis AI Assistant

## 🎯 Project Objective

### Project Structure

```text
llm-DSL-info-extraction/
├── agents/                 # AI agents package
│   ├── __init__.py
│   ├── base.py            # LLMAgent abstract interface
│   ├── gpt_agent.py       # OpenAI implementation
│   ├── mistral_agent.py   # Mistral implementation
│   └── gemini_agent.py    # Google Gemini implementation
├── .env.example           # Configuration template
├── .gitignore            # Files ignored by git
├── test.py               # Model testing script
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

This project aims to create an advanced system to assist **LOKAD**'s *Supply Chain Scientists*. The goal is to help them navigate, understand, and query a complex codebase written in **Envision**, LOKAD's proprietary language, using natural language questions.

* **Client**: [LOKAD](https://www.lokad.com)
* **Framework**: Collective Scientific Project (PSC) - École Polytechnique (X24)

## 💻 Prerequisites

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

On Windows:

```bash
python -m venv env
.\env\Scripts\activate
```

On Unix/macOS:

```bash
python -m venv env
source env/bin/activate
```

### 3. Installing Dependencies

Update pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration des API Keys

1. Copy the configuration file:

```bash
# On Windows
copy .env.example .env

# On Unix/macOS
cp .env.example .env
```

2. Configure your API keys:
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

## 🔄 Pipeline Architecture

Here's the target architecture that will be developed:

### 1. 📚 Knowledge Preparation Flow (Offline)

This flow prepares the knowledge base from raw Envision scripts.

* **`Data Base`**: The collection of `.nvm` scripts provided by LOKAD.
* **`Parser`**: An intelligent module that analyzes scripts, segments them into logical chunks, and extracts metadata (functions, variables, etc.).
* **`RAG (Retrieval Augmented Generation)`**: A vector database (e.g., FAISS) that indexes chunks and metadata for fast semantic search.

### 2. 💡 Execution and Evaluation Flow (Online)

This flow handles user requests and generates validated responses.

* **`Question`**: The user's question in natural language.
* **`Engineered Prompt`**: An "augmented" prompt that combines the question with the most relevant code chunks retrieved by the `RAG`.
* **`Main LLM`**: The main language model (e.g., GPT-4) that generates a response from the prompt.
* **`Logic Checker`**: A verification module that checks the syntax and logical consistency of the response. It enables a **correction loop** in case of errors.
* **`Final Answer`**: The final response, validated and presented to the user.
* **`Answer Grader (Scorer)`** : Un évaluateur qui note la qualité de la réponse finale en la comparant à une réponse de référence (utilisé pour le benchmark).
