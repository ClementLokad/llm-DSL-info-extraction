from agents.gemini_agent import GeminiAgent
from agents.mistral_agent import MistralAgent
from agents.gpt_agent import GPTAgent
from agents.llama3_agent import Llama3Agent
from agents.groq_agent import GroqAgent
from agents.qwen_agent import QwenAgent
from agents.base import LLMAgent
from config_manager import get_config

def prepare_agent(agent_name: str) -> LLMAgent:
        """Prepare and return the default LLM agent based on configuration."""
        agent : LLMAgent
        if agent_name == 'gemini':
            agent = GeminiAgent()
        elif agent_name == 'mistral':
            agent = MistralAgent()
        elif agent_name == 'gpt':
            agent = GPTAgent()
        elif agent_name == 'llama3':
            agent = Llama3Agent()
            agent = GPTAgent()
        elif agent_name == 'llama3.2':
            agent = Llama3Agent('llama3.2')
        elif agent_name == 'qwen':
            agent = QwenAgent()
        elif agent_name == 'groq':
            agent = GroqAgent()
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")
        agent.initialize()
        return agent

def prepare_default_agent() -> LLMAgent:
    """Prepare and return the default LLM agent based on configuration."""
    return prepare_agent(get_config().get_default_agent().lower())

def prepare_benchmark_agent() -> LLMAgent:
    """Prepare and return the benchmark LLM agent based on configuration."""
    return prepare_agent(get_config().get_benchmark_agent().lower())

def prepare_summary_agent() -> LLMAgent:
    """Prepare and return the embedder summary LLM agent based on configuration."""
    return prepare_agent(get_config().get_summary_agent().lower())

def prepare_query_transformer_agent() -> LLMAgent:
    """Prepare and return the query transformer LLM agent based on configuration."""
    return prepare_agent(get_config().get_query_transformer_agent().lower())