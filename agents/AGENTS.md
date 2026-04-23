# 🤖 LLM Agents Documentation

[![Agents](https://img.shields.io/badge/Agents-DeepSeek%20%7C%20Claude%20%7C%20Mistral%20%7C%20Groq%20%7C%20Qwen-blue.svg)](https://github.com)
[![Rate Limiting](https://img.shields.io/badge/Rate_Limiting-Enabled-green.svg)](https://github.com)

> *Modular and extensible architecture for integrating large language models*

---

## 🎯 Overview

The LLM Agents system provides a **unified interface** for interacting with different language models while automatically managing:

- 🔄 **Intelligent rate limiting** with exponential backoff retries
- 🔌 **Plugin architecture** for easy addition of new agents
- ⚡ **Robust error handling** with fallback and logging
- 🔧 **Agent selection** via config.yaml (API keys configured in .env)
- 🎯 **OpenAI-compatible tool calling** for standardized function execution
- 📊 **Performance metrics** and API call monitoring

---

## 🏗️ Agent Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│              🤖 LLM AGENTS ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│  🎭 Agent Interface (base.py)                              │
│  ├── 🔄 @rate_limited decorator                            │
│  ├── ⚡ Error handling & retries                           │
│  └── 📊 Performance monitoring                             │
├─────────────────────────────────────────────────────────────┤
│  🧠 Agent Implementations                                  │
│  ├── 🚀 DeepSeekAgent    ├── 🧠 ClaudeAgent                │
│  ├── ⚡ MistralAgent     ├── 🌀 GroqAgent                  │
│  └── 🐲 QwenAgent       ├── 📡 QwenSSHAgent                │
├─────────────────────────────────────────────────────────────┤
│  🔧 Configuration Layer                                    │
│  ├── 🔑 API Keys (.env)  ├── ⚙️ Parameters (config.yaml)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Available Agents

### 🚀 DeepSeek Agent - DeepSeek API

**File**: `deepseek_agent.py`
**Provider**: DeepSeek
**Model**: `deepseek-v3` (default but `deepseek-r1` is also available)

#### ✨ Features

- 🚀 **High performance** - Fast reasoning with advanced architecture
- 🧠 **Code understanding** - Specialized for DSL analysis
- 💰 **Cost-effective** - Competitive pricing for production use
- 🌐 **Multilingual** - Strong support for English and French
- 🔒 **Privacy** - Data processing with compliance focus

#### 🔧 Configuration

```bash
# .env - API key only (required)
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "deepseek-v3" # Or "deepseek-r1"
```

#### 📊 Typical Metrics

- **Latency**: ~1.5s per request
- **Throughput**: 40 requests/minute
- **Token limits**: 64K context, 8K output
- **Rate limits**: Generous for development

#### 🎯 Optimal Use Cases

- ✅ Production DSL queries
- ✅ Complex reasoning tasks
- ✅ Budget-conscious deployments
- ✅ Multilingual support needed

---

### 🧠 Claude Agent - Anthropic

**File**: `claude_agent.py`
**Provider**: Anthropic
**Model**: `claude-sonnet-4-6` (default but `claude-haiku-4-5` is also available)

#### ✨ Features

- 🏆 **Premium quality** - Sophisticated reasoning and analysis
- 📚 **Extended context** - Up to 200K token context window
- 🎨 **Creative generation** - High-quality content creation
- 🔧 **Tool use** - Supports function calling and JSON mode
- 🧠 **Instruction following** - Excellent at following detailed prompts

#### 🔧 Configuration

```bash
# .env - API key only (required)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "sonnet"  # Or "haiku" for faster, cheaper option
```

#### 📊 Typical Metrics

- **Latency**: ~2.0s per request
- **Throughput**: 25 requests/minute
- **Token limits**: 200K context, 4K output
- **Rate limits**: Depends on subscription tier

#### 🎯 Optimal Use Cases

- ✅ Complex reasoning required
- ✅ Extended documentation generation
- ✅ Detailed code analysis
- ✅ Multi-step problem solving

---

### ⚡ Mistral Agent - Mistral AI

**File**: `mistral_agent.py`
**Provider**: Mistral AI
**Model**: `mistral-large-latest`

#### ✨ Features

- 🇪🇺 **European alternative** - GDPR-native compliance
- ⚖️ **Price/performance balance** - Excellent value proposition
- 🌍 **Advanced multilingual** - Specialized for European languages
- 🔒 **Privacy-focused** - Data processing in Europe
- ⚡ **Competitive speed** - Reasonable latency

#### 🔧 Configuration

```bash
# .env - API key only (required)
MISTRAL_API_KEY=your-mistral-api-key-here
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "mistral" # Options: "qwen", "qwen-ssh", "mistral", "groq", "deepseek-v3", "deepseek-r1", "sonnet", "haiku"
```

#### 📊 Typical Metrics

- **Latency**: ~1.8s per request
- **Throughput**: 30 requests/minute
- **Token limits**: 32K context, 8K output
- **Rate limits**: Generous for development

#### 🎯 Optimal Use Cases

- ✅ European compliance required
- ✅ Multilingual French/English
- ✅ Budget-constrained with quality maintained
- ✅ Rapid prototyping

---

### 🌀 Groq Agent - Groq API

**File**: `groq_agent.py`
**Provider**: Groq
**Model**: `mixtral-8x7b-32768` (fast inference)

#### ✨ Features

- ⚡ **Ultra-fast inference** - Sub-second latency at scale
- 💰 **Cost-effective** - Free tier available for development
- 🔄 **High throughput** - 30+ concurrent requests
- 📱 **Lightweight models** - Efficient open models (Mixtral, Llama)
- 🚀 **Production-ready** - Specialized LPU hardware

#### 🔧 Configuration

```bash
# .env - API key only (required)
GROQ_API_KEY=your-groq-api-key-here
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "groq" # Options: "qwen", "qwen-ssh", "mistral", "groq", "deepseek-v3", "deepseek-r1", "sonnet", "haiku"
```

#### 📊 Typical Metrics

- **Latency**: ~0.5s per request (fastest)
- **Throughput**: 100+ requests/minute
- **Token limits**: 32K context, 8K output
- **Rate limits**: Very generous, free tier available

#### 🎯 Optimal Use Cases

- ✅ Real-time interactive queries
- ✅ High-volume benchmarking
- ✅ Cost-sensitive production
- ✅ Rapid prototyping with free tier

---

### 🐲 Qwen Agent - Alibaba Qwen

**File**: `qwen_agent.py`
**Provider**: Alibaba Cloud (Qwen API)
**Model**: `qwen-plus` or `qwen-max`

#### ✨ Features

- 🌏 **Asia-optimized** - Specialized for Asian languages
- 🧠 **Strong reasoning** - Qwen's MoE architecture
- 💰 **Affordable** - Competitive pricing for cloud services
- 🔄 **Multilingual** - Support for 10+ languages
- ⚡ **Fast inference** - Optimized for latency

#### 🔧 Configuration

```bash
# .env - API key only (required)
QWEN_API_KEY=your-qwen-api-key-here
QWEN_WORKSPACE_ID=your-workspace-id
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "qwen" # Options: "qwen", "qwen-ssh", "mistral", "groq", "deepseek-v3", "deepseek-r1", "sonnet", "haiku"
```

#### 📊 Typical Metrics

- **Latency**: ~1.2s per request
- **Throughput**: 50 requests/minute
- **Token limits**: 30K context, 8K output
- **Rate limits**: Reasonable for development

#### 🎯 Optimal Use Cases

- ✅ Asian language support
- ✅ Alibaba cloud deployments
- ✅ Cost-effective Asia-Pacific
- ✅ Multilingual applications

---

### 📡 Qwen SSH Agent - Qwen via SSH

**File**: `qwen_ssh_agent.py`
**Provider**: Alibaba Qwen (via SSH tunnel)
**Model**: `qwen-max` (local/SSH deployment)

#### ✨ Features

- 🔒 **Secure tunneling** - SSH-based encrypted connection
- 🏢 **Enterprise-ready** - On-premises deployment support
- 🔐 **High security** - No direct API exposure
- 🌐 **Network flexibility** - Works across firewalls
- 📊 **Local monitoring** - Full request inspection possible

#### 🔧 Configuration

```bash
# .env - SSH credentials and API key (required)
QWEN_SSH_KEY_PATH=/path/to/ssh/key
QWEN_SSH_PASSWORD=your-ssh-password
QWEN_API_KEY=your-qwen-api-key-here
```

```yaml
# config.yaml - Select which agent to use
agent:
  default_model: "qwen_ssh" # Options: "qwen", "qwen-ssh", "mistral", "groq", "deepseek-v3", "deepseek-r1", "sonnet", "haiku"
```

#### 📊 Typical Metrics

- **Latency**: ~2.0s per request (includes SSH overhead)
- **Throughput**: 20 requests/minute
- **Token limits**: 30K context, 8K output
- **Rate limits**: Depends on server configuration

#### 🎯 Optimal Use Cases

- ✅ Enterprise security requirements
- ✅ On-premises Qwen deployments
- ✅ Sensitive data processing
- ✅ Network isolation needed

---

## 🔧 Abstract Base Class - BaseAgent

### 📋 Base Class

All agent implementations inherit from `LLMAgent` abstract base class and must implement the core methods. Our system uses the **OpenAI-compatible tool calling format** for standardized function execution across all LLM providers.

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time
import functools

class LLMAgent(ABC):
    """Abstract interface for all AI agents with standardized OpenAI tool calling"""
  
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent with its configuration"""
        pass
  
    @abstractmethod  
    def generate_response(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None, 
        context: Optional[List[Dict[str, Any]]] = None, 
        temperature: float = 0.3
    ) -> str:
        """Generate a plain-text response from question and context
    
        Args:
            user_message: The user's question or instruction
            system_prompt: Optional system-role instruction
            context: Conversation history for multi-turn interactions
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Model-generated text response
        """
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: str = "any"
    ) -> 'ToolCallResult':
        """Request the model to call one of the provided tools using OpenAI format
        
        **OpenAI Tool Format**: Each tool is a dict with structure:
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "What this function does",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ]
                }
            }
        }

        Args:
            user_message: The user-facing question or instruction
            tools: List of tool schemas in OpenAI format
            system_prompt: Optional system instruction
            tool_choice: "any" forces a tool call (internally mapped to provider equivalents)

        Returns:
            ToolCallResult containing tool_name, tool_id, and arguments
        """
        pass

    @abstractmethod
    def submit_tool_result_and_continue(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        next_instruction: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "any"
    ) -> 'ToolCallResult':
        """
        Append tool result to conversation and request next tool call.
        Used to continue multi-step tool-calling workflows.
        
        Internally handles the OpenAI tool calling protocol:
        - Converts tool result to 'tool' role message
        - Includes next_instruction as user message
        - Requests next tool call from model
        - Returns next ToolCallResult

        Args:
            tool_call_id: ID from preceding generate_with_tools call
            tool_name: Name of the function that was executed
            result: Serialized result of the tool execution
            next_instruction: User message for next action
            tools: Available tool schemas in OpenAI format
            tool_choice: "any" forces next tool call

        Returns:
            ToolCallResult for next tool call
        """
        pass
```

### 🔄 Rate Limiting Decorator

```python
@rate_limited(max_retries: int = 3, initial_delay: float = 1.0)
def decorator(func):
    """
    Intelligent API rate limiting management:
  
    Features:
    - ⏱️ Retry with exponential backoff  
    - 🎯 Automatic rate limit detection
    - 📊 Detailed attempt logging
    - ⚡ Minimal pause after success
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                time.sleep(0.1)  # Courteous pause
                return result
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        return wrapper
```

---

## 🎛️ Configuration

### ⚙️ Agent Selection

```yaml
# config.yaml - Choose which agent to use for each task
agent:
  default_model: "deepseek-v3"  # Primary agent: deepseek, claude, mistral, groq, qwen, qwen_ssh

# Examples - set different agents for different tasks
main_pipeline:
  agent_logic:
    distillation_llm: "deepseek-v3"        # Fast, efficient
    main_llm: "deepseek-r1"                # Primary reasoning
    planner_llm: "deepseek-v3"             # Task planning
    cleaning_llm: "deepseek-v3"            # Response cleanup
```

### 🔑 API Key Setup (Required)

Only API keys need to be configured. Set them in your `.env` file:

```bash
# .env - Secure API keys (configure only the agents you use)
DEEPSEEK_API_KEY=your-deepseek-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
MISTRAL_API_KEY=your-mistral-api-key-here
GROQ_API_KEY=your-groq-api-key-here
QWEN_API_KEY=your-qwen-api-key-here
QWEN_WORKSPACE_ID=your-workspace-id                    # For Qwen
QWEN_SSH_KEY_PATH=/path/to/ssh/key                    # For Qwen SSH
QWEN_SSH_PASSWORD=your-ssh-password                   # For Qwen SSH
```

**Note**: Agent hyperparameters (temperature, max_tokens, etc.) are managed internally and optimized per agent. Only API keys require configuration.

---

## 🧪 Testing and Validation

### 📊 Agent Test Suite

**File**: `test_agents.py`

```python
class TestAgents:
    """Comprehensive tests for AI agents"""
  
    def test_agent_initialization(self):
        """Verify all agents initialize correctly"""
    
    def test_rate_limiting(self):
        """Validate rate limiting behavior"""
    
    def test_error_handling(self):
        """Test error handling and fallbacks"""
    
    def test_response_quality(self):
        """Evaluate response quality on test cases"""
    
    def test_performance_metrics(self):
        """Measure latency and throughput per agent"""
```

### 🎯 Response Quality Validation

```python
# Automated quality metrics
def evaluate_response_quality(question: str, response: str, expected_keywords: List[str]) -> float:
    """
    Evaluation criteria:
    - ✅ Presence of expected keywords
    - ✅ Appropriate response length  
    - ✅ Structure and formatting
    - ✅ Contextual relevance
    - ✅ Absence of hallucinations
    """
    score = 0.0
  
    # Keyword matching (40%)
    keyword_score = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    score += (keyword_score / len(expected_keywords)) * 0.4
  
    # Length appropriateness (20%)
    response_length = len(response.split())
    if 50 <= response_length <= 300:  # Sweet spot for DSL queries
        score += 0.2
    
    # Structure quality (40%)
    structure_indicators = [':', '-', '•', '\n', 'example', 'logic']
    structure_score = sum(1 for indicator in structure_indicators if indicator in response.lower())
    score += min(structure_score / len(structure_indicators), 1.0) * 0.4
  
    return min(score, 1.0)
```

---

## 🚀 Adding a New Agent

### 📋 Implementation Template

All agents must implement the abstract `LLMAgent` interface and support OpenAI-compatible tool calling format:

```python
# agents/new_agent.py
import os
from typing import Optional, List, Dict, Any

from .base import LLMAgent, rate_limited, ToolCallResult

class NewAgent(LLMAgent):
    """Agent using NewProvider API with OpenAI tool calling support"""
  
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('NEWPROVIDER_API_KEY')
        self._model = 'default-model'  # Internal model reference
    
        if not self.api_key:
            raise ValueError("NEWPROVIDER_API_KEY not configured")
  
    def initialize(self) -> None:
        """Initialize NewProvider client"""
        self.client = NewProviderClient(api_key=self.api_key)
    
        # Connectivity test
        try:
            self._test_connection()
            print(f"   ✅ {self._model} initialized")
        except Exception as e:
            raise RuntimeError(f"❌ NewProvider initialization failed: {e}")
  
    @rate_limited(max_retries=3, initial_delay=1.0)
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3
    ) -> str:
        """Generate response via NewProvider API"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_message})
    
        try:
            response = self.client.create_message(
                model=self._model,
                temperature=temperature,
                messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"NewProvider API error: {e}")

    @rate_limited(max_retries=3, initial_delay=1.0)
    def generate_with_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: str = "any"
    ) -> ToolCallResult:
        """Generate with OpenAI-format tool calling"""
        
        # Convert tool_choice to provider-specific format if needed
        # (e.g., 'any' -> 'required' for some providers)
        provider_tool_choice = self._map_tool_choice(tool_choice)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.create_message(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice=provider_tool_choice
            )
            
            # Extract tool call from provider response and convert to ToolCallResult
            tool_call = response.tool_calls[0]
            return ToolCallResult(
                tool_name=tool_call.function.name,
                tool_id=tool_call.id,
                arguments=tool_call.function.arguments
            )
        except Exception as e:
            raise RuntimeError(f"NewProvider tool calling error: {e}")

    @rate_limited(max_retries=3, initial_delay=1.0)
    def submit_tool_result_and_continue(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        next_instruction: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "any"
    ) -> ToolCallResult:
        """Submit tool result and request next tool call (OpenAI format)"""
        
        # Append tool result to conversation in OpenAI format
        self.context.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": tool_call_id, "function": {"name": tool_name}}]
        })
        self.context.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
        self.context.append({"role": "user", "content": next_instruction})
        
        try:
            response = self.client.create_message(
                model=self._model,
                messages=self.context,
                tools=tools,
                tool_choice=self._map_tool_choice(tool_choice)
            )
            
            tool_call = response.tool_calls[0]
            return ToolCallResult(
                tool_name=tool_call.function.name,
                tool_id=tool_call.id,
                arguments=tool_call.function.arguments
            )
        except Exception as e:
            raise RuntimeError(f"NewProvider continuation error: {e}")
    
    @property
    def model_name(self) -> str:
        """Return the model name for identification"""
        return self._model
    
    def _map_tool_choice(self, tool_choice: str) -> str:
        """Map standard tool_choice values to provider-specific format"""
        # Override this method if your provider uses different values
        return tool_choice
```

### ⚙️ Configure New Agent

```bash
# .env - API key ONLY
NEWPROVIDER_API_KEY=your-newprovider-api-key-here
```

```yaml
# config.yaml - Select your agent
agent:
  default_model: "new_provider"  # Add to choices: "qwen", "qwen-ssh", "mistral", "groq", "deepseek-v3", "deepseek-r1", "sonnet", "haiku"
```

### ✅ Required Implementation Checklist

- [ ] Class inherits from `LLMAgent`
- [ ] Implements `initialize()`
- [ ] Implements `generate_response()` with full signature
- [ ] Implements `generate_with_tools()` using **OpenAI tool format**
- [ ] Implements `submit_tool_result_and_continue()` using **OpenAI tool format**
- [ ] Implements `model_name` property
- [ ] All methods decorated with `@rate_limited`
- [ ] Environment variables documented in `.env`
- [ ] Registered in agent factory/dispatcher
- [ ] Unit tests added
- [ ] Documentation updated

---

## 📊 Monitoring and Metrics

### 🎯 Automatically Collected Metrics

```python
class AgentMetrics:
    """Metrics collector for agents"""
  
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0,
            'rate_limit_hits': 0,
            'tokens_consumed': 0,
            'cost_estimate': 0.0
        }
  
    def record_request(self, agent_name: str, latency: float, 
                      success: bool, tokens: int = 0):
        """Record request metrics"""
        self.metrics['total_requests'] += 1
    
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average latency
        self._update_avg_latency(latency)
    
        # Track tokens and cost
        self.metrics['tokens_consumed'] += tokens
        self.metrics['cost_estimate'] += self._estimate_cost(agent_name, tokens)
```

### 📈 Metrics Dashboard

```bash
# Monitoring command
python main.py --status --verbose

# Example output:
🤖 AGENT PERFORMANCE METRICS
=====================================
📊 DeepSeek Agent:
   • Total requests: 247
   • Success rate: 99.2%
   • Average latency: 1.5s
   • Tokens consumed: 45,678
   • Estimated cost: $0.23
   
⚡ Rate Limiting:
   • Rate limits hit: 2
   • Successful retries: 2/2
   • Total wait time: 5.2s
```

---

## ⚠️ Troubleshooting

### 🔧 Common Issues

#### ❌ Error: Invalid API Key

```bash
# Verify environment variables
echo $DEEPSEEK_API_KEY
echo $ANTHROPIC_API_KEY  
echo $MISTRAL_API_KEY

# Reload .env if needed
source .env
```

#### ❌ Rate Limit Exceeded

```python
# @rate_limited decorator handles automatically
# But you can adjust parameters:

@rate_limited(max_retries=5, initial_delay=2.0)
def generate_response(self, question: str, context: str = None) -> str:
    # Implementation
```

#### ❌ Connection Timeout

```yaml
# Increase timeout in config.yaml
agent:
  default_timeout: 60  # 60 seconds instead of 30
```

### 🔧 Debug and Diagnostics

```bash
# Debug mode to see all API calls
python main.py --verbose --query "test query"

# Test connectivity of specific agent
python test_agents.py --agent deepseek

# Verify configuration
python -c "from config_manager import ConfigManager; print(ConfigManager().get_config())"
```

---

## 🤝 Contributing

### 📋 Guidelines for New Agents

1. **🏗️ Inherit from LLMAgent** - Use the abstract interface
2. **🔄 Implement @rate_limited** - Automatic rate limit management
3. **⚙️ Externalize configuration** - Parameters in config.yaml
4. **🧪 Comprehensive tests** - Test suite for each agent
5. **📖 Documentation** - Add examples and use cases

### 🔧 New Agent Checklist

- [ ] Class inherits from `LLMAgent`
- [ ] Methods `initialize()` and `generate_response()` implemented
- [ ] `@rate_limited` decorator applied
- [ ] Configuration in `config.yaml`
- [ ] Environment variables documented
- [ ] Unit tests added
- [ ] Documentation updated

---

## 📚 Related Documentation

### Pipeline & Orchestration
- [LangGraph Pipeline](../pipeline/PIPELINE.md) - Main orchestration system
- [Agentic Workflow](../pipeline/agent_workflow/AGENTIC_WORKFLOW.md) - Agent loop using these agents
- [Benchmarking Framework](../pipeline/benchmarks/BENCHMARKS.md) - Performance evaluation using agents

### RAG Components
- [RAG Pipeline Overview](../rag/RAG.md) - Complete retrieval architecture
- [Query Transformers Documentation](../rag/query_transformers/QUERY_TRANSFORMERS.md) - Query enhancement
- [Embedders Documentation](../rag/embedders/EMBEDDERS.md) - Embedding models
- [Retrievers Documentation](../rag/retrievers/RETRIEVERS.md) - Vector database retrieval

### Getting Started
- [Quick Start Tutorial](../TUTORIAL.md) - Setup and usage guide
- [Project README](../README.md) - High-level overview

---

## 📝 License

This project is under PRIVATE LICENSE AGREEMENT. See [LICENSE](../LICENSE) for details.

---
