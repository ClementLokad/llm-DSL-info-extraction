from agents.gemini_agent import GeminiAgent
from agents.mistral_agent import MistralAgent
from agents.gpt_agent import GPTAgent
from agents.llama3_agent import Llama3Agent
from agents.groq_agent import GroqAgent
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

def prepare_embedder_summary_agent() -> LLMAgent: #OLD BACKUP FUNCTION
    """Prepare and return the embedder summary LLM agent based on configuration."""
    return prepare_agent(get_config().get_embedder_summary_agent().lower())

def prepare_chunk_summary_agent() -> LLMAgent:
    """Prepare and return the embedder summary LLM agent based on configuration."""
    return prepare_agent(get_config().get_chunk_summary_agent().lower())