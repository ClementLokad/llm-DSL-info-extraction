# 🚀 Phase 2: Query & Response System - Development Plan

## 📋 **Phase 2 Overview**

With preprocessing complete, we now build a comprehensive **RAG (Retrieval-Augmented Generation)** system that:

1. **Accepts natural language queries** about Envision DSL code
2. **Retrieves relevant code chunks** using vector similarity
3. **Generates intelligent answers** using LLM agents
4. **Provides code examples** and explanations

---

## 🏗️ **Architecture Design**

### **Core Components to Build:**

#### 1. **Query Processing Module** (`query/`)
```
query/
├── __init__.py
├── parser.py          # Natural language query parsing
├── intent_classifier.py # Classify query type (code search, explanation, examples)
├── context_builder.py # Build context from retrieved chunks
└── validator.py       # Validate and sanitize queries
```

#### 2. **Retrieval System** (`retrieval/`)
```
retrieval/
├── __init__.py
├── vector_search.py   # Enhanced vector similarity search
├── hybrid_search.py   # Combine vector + keyword search
├── ranking.py         # Re-rank results by relevance
└── filter.py          # Filter by file types, patterns, etc.
```

#### 3. **Response Generation** (`response/`)
```
response/
├── __init__.py
├── generator.py       # Main response generation logic
├── prompt_templates.py # LLM prompt engineering
├── code_formatter.py  # Format code examples nicely
└── answer_validator.py # Validate LLM responses
```

#### 4. **Interactive Interface** (`interface/`)
```
interface/
├── __init__.py
├── cli.py            # Command-line interface
├── web_app.py        # Optional: FastAPI web interface
└── chat_session.py   # Maintain conversation context
```

---

## 🎯 **Development Priorities**

### **Priority 1: Core RAG Pipeline** (Week 1-2)

#### **Task 1.1: Enhanced Vector Search**
- **File**: `retrieval/vector_search.py`
- **Purpose**: Upgrade existing search with better ranking
- **Features**:
  - Semantic similarity scoring
  - Multi-vector search (combine different embeddings)
  - Result diversity (avoid duplicate patterns)
  - Configurable result limits

#### **Task 1.2: Query Intent Classification**
- **File**: `query/intent_classifier.py`
- **Purpose**: Understand what user wants
- **Intent Types**:
  - `CODE_SEARCH`: "Find functions that calculate profit"
  - `EXPLANATION`: "How does this code work?"
  - `EXAMPLES`: "Show me examples of table definitions"
  - `DEBUGGING`: "What's wrong with this code?"
  - `PATTERNS`: "What are common patterns for..."

#### **Task 1.3: Context Building**
- **File**: `query/context_builder.py`
- **Purpose**: Prepare relevant context for LLM
- **Features**:
  - Chunk deduplication
  - Context window management (stay under token limits)
  - Related code inclusion (if chunk references other parts)
  - Metadata enrichment (file paths, code types)

### **Priority 2: Response Generation** (Week 2-3)

#### **Task 2.1: Prompt Engineering**
- **File**: `response/prompt_templates.py`
- **Purpose**: Create effective LLM prompts
- **Templates**:
  ```python
  ENVISION_EXPERT_PROMPT = """
  You are an expert in LOKAD Envision DSL. Based on the provided code chunks:
  
  Query: {query}
  
  Relevant Code:
  {context}
  
  Provide a clear answer with:
  1. Direct answer to the question
  2. Code examples if relevant
  3. Best practices recommendations
  """
  ```

#### **Task 2.2: Response Generator**
- **File**: `response/generator.py`
- **Purpose**: Main LLM integration
- **Features**:
  - Support multiple LLM providers (reuse agents/)
  - Stream responses for better UX
  - Include confidence scores
  - Handle API errors gracefully

### **Priority 3: User Interface** (Week 3-4)

#### **Task 3.1: CLI Interface**
- **File**: `interface/cli.py`
- **Purpose**: Interactive command-line chat
- **Features**:
  ```bash
  python chat.py "How do I calculate profit margins in Envision?"
  python chat.py --interactive  # Chat mode
  python chat.py --file query.txt  # Process file of queries
  ```

#### **Task 3.2: Chat Session Management**
- **File**: `interface/chat_session.py`
- **Purpose**: Maintain conversation context
- **Features**:
  - Remember previous questions
  - Context carryover between queries
  - Session history and replay

---

## 🔧 **Technical Implementation Details**

### **Integration with Existing System**
```python
# Main application structure
from agents import get_embedder
from preprocessing.pipeline import PreprocessingPipeline
from retrieval.vector_search import VectorSearch
from response.generator import ResponseGenerator

class EnvisionRAG:
    def __init__(self, model_name="gemini"):
        self.embedder = get_embedder(model_name)
        self.vector_search = VectorSearch(embedder=self.embedder)
        self.response_generator = ResponseGenerator(model_name)
    
    async def query(self, question: str) -> str:
        # 1. Process query
        intent = classify_intent(question)
        
        # 2. Retrieve relevant chunks  
        chunks = await self.vector_search.search(question, limit=10)
        
        # 3. Build context
        context = build_context(chunks, intent)
        
        # 4. Generate response
        response = await self.response_generator.generate(
            question, context, intent
        )
        
        return response
```

### **Configuration Updates**
Add to `.env`:
```env
# RAG Configuration
RAG_MODEL=gemini
MAX_CONTEXT_TOKENS=4000
SEARCH_RESULTS_LIMIT=10
ENABLE_HYBRID_SEARCH=true
RESPONSE_TEMPERATURE=0.1
```

---

## 📊 **Success Metrics**

### **Phase 2 Success Criteria:**
1. **Query Response Time**: < 3 seconds per query
2. **Answer Relevance**: User can find relevant code 90% of time
3. **Code Example Quality**: Generated examples are syntactically correct
4. **Conversation Flow**: Multi-turn conversations work naturally
5. **Error Handling**: Graceful handling of edge cases

### **Testing Strategy:**
- **Unit Tests**: Each component individually
- **Integration Tests**: Full RAG pipeline
- **User Testing**: Manual evaluation of answer quality
- **Performance Tests**: Response time and throughput

---

## 🎯 **Sample Use Cases to Implement**

### **Use Case 1: Code Search**
```
Query: "Find all code that calculates total revenue"
Expected: Return chunks with revenue calculations + explanations
```

### **Use Case 2: Code Explanation**
```
Query: "Explain how this profit calculation works: TotalProfit = Revenue - Costs"
Expected: Break down the logic, explain variables, show context
```

### **Use Case 3: Pattern Discovery**
```
Query: "What are common patterns for reading CSV files?"
Expected: Show multiple examples of read statements with explanations
```

### **Use Case 4: Debugging Help**
```
Query: "Why might this code fail: Orders.Price where Orders.Quantity > 0"
Expected: Identify potential issues, suggest fixes
```

---

## 🚀 **Next Immediate Actions**

### **Week 1 Tasks:**
1. **Set up Phase 2 directories** (`query/`, `retrieval/`, `response/`, `interface/`)
2. **Implement enhanced vector search** with ranking improvements
3. **Build query intent classifier** using pattern matching or lightweight ML
4. **Create basic prompt templates** for Envision DSL expert responses

### **Week 2 Milestones:**
- ✅ **Working RAG pipeline**: Question → Retrieval → Response
- ✅ **CLI interface**: Basic chat functionality
- ✅ **Response quality**: Meaningful answers to simple queries

### **Development Commands:**
```bash
# Set up new structure
mkdir query retrieval response interface

# Test the RAG system
python -m pytest tests/test_rag.py

# Run interactive chat
python chat.py --interactive

# Benchmark performance
python benchmark_rag.py
```

---

## 🎉 **Expected Outcomes**

By completing Phase 2, you'll have:

1. **🔍 Intelligent Code Search**: Natural language queries return relevant Envision code
2. **💡 Expert Explanations**: AI explains complex DSL patterns and logic
3. **📚 Learning Assistant**: Users can learn Envision best practices through examples
4. **🐛 Debugging Helper**: AI identifies potential issues and suggests fixes
5. **🚀 Production RAG System**: Scalable architecture for enterprise deployment

**Ready to transform your Envision codebase into an intelligent, queryable knowledge base!** 🌟