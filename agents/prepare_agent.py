from agents.claude_agent import ClaudeAgent
from agents.mistral_agent import MistralAgent
from agents.groq_agent import GroqAgent
from agents.qwen_agent import QwenAgent
from agents.qwen_ssh_agent import QwenSSHAgent
from agents.deepseek_agent import DeepSeekAgent
from agents.base import LLMAgent
from utils.config_manager import get_config

def prepare_agent(agent_name: str) -> LLMAgent:
        """Prepare and return the default LLM agent based on configuration."""
        agent : LLMAgent
        if agent_name == 'mistral':
            agent = MistralAgent()
        elif agent_name == 'qwen':
            agent = QwenAgent()
        elif agent_name == 'qwen-ssh':
            agent = QwenSSHAgent()
        elif agent_name == 'groq':
            agent = GroqAgent()
        elif agent_name == 'deepseek-v3':
            agent = DeepSeekAgent(model="deepseek-chat")
        elif agent_name == 'deepseek-r1':
            agent = DeepSeekAgent(model="deepseek-reasoner")
        elif agent_name == 'sonnet':
            agent = ClaudeAgent(model="claude-sonnet-4-6")
        elif agent_name == 'haiku':
            agent = ClaudeAgent(model="claude-haiku-4-5")
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