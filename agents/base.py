from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple
from config_manager import get_config
from rag.utils.handle_tokens import get_token_count
from dataclasses import dataclass
import time
import functools


@dataclass
class ToolCallResult:
    tool_name: str
    tool_id: str        # providers without IDs can use a uuid4()
    arguments: dict


def rate_limited(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator to handle rate limiting and retries for LLM API calls.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(0.1)
                    return result
                except Exception as e:
                    error_msg = str(e).lower()
                    if ("rate limit" in error_msg or "quota" in error_msg) and attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        print(f"Rate limit hit. Waiting {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        continue
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed after {max_retries} retries: {str(e)}")
                    raise
        return wrapper
    return decorator


class LLMAgent(ABC):
    """Abstract interface for LLM agents."""
    def __init__(self):
        self.context: List[Dict[str, Any]] = []  # Conversation history and system prompts

    @staticmethod
    def count_tokens(func):
        """
        Decorator that counts input and output tokens for any LLM call.

        Input counting strategy per method:
          generate_response              → user_message + system_prompt (if any)
          generate_with_tools            → user_message + system_prompt (if any)
          submit_tool_result_and_continue→ result + next_instruction

        All parameters may arrive as positional args or keyword args depending
        on the call site, so we resolve them by name from the function signature
        using functools rather than relying on positional index.
        """
        import inspect
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())  # includes 'self'

        @functools.wraps(func)
        def wrapper(agent, *args, **kwargs):
            if not get_config().get("main_pipeline.token_count", False):
                return func(agent, *args, **kwargs)

            # Build a unified name→value mapping for all arguments
            # (positional args are matched to parameter names by position,
            #  skipping 'self' which is already bound to agent)
            bound: Dict[str, Any] = {}
            for i, val in enumerate(args):
                # param_names[0] is 'self', positional args start at index 1
                if i + 1 < len(param_names):
                    bound[param_names[i + 1]] = val
            bound.update(kwargs)

            # --- Count input tokens ---
            tokens_in = 0

            # user_message is present in generate_response and generate_with_tools
            user_message = bound.get("user_message", "")
            if isinstance(user_message, str):
                tokens_in += get_token_count(user_message)

            # system_prompt is optional in generate_response and generate_with_tools
            system_prompt = bound.get("system_prompt", "")
            if isinstance(system_prompt, str) and system_prompt:
                tokens_in += get_token_count(system_prompt)

            # result + next_instruction are present in submit_tool_result_and_continue
            tool_result = bound.get("result", "")
            if isinstance(tool_result, str) and tool_result:
                tokens_in += get_token_count(tool_result)

            next_instruction = bound.get("next_instruction", "")
            if isinstance(next_instruction, str) and next_instruction:
                tokens_in += get_token_count(next_instruction)
            
            try:
                get_config().config["tokens_in"] += tokens_in
            except KeyError:
                get_config().config["tokens_in"] = tokens_in

            # --- Call the actual method ---
            res = func(agent, *args, **kwargs)

            # --- Count output tokens ---
            # generate_response returns a str
            # generate_with_tools and submit_tool_result_and_continue return ToolCallResult
            get_config().config.setdefault("tokens_out", 0)
            if isinstance(res, str):
                get_config().config["tokens_out"] += get_token_count(res)
            elif isinstance(res, ToolCallResult):
                # Count the stringified arguments as a proxy for output tokens
                get_config().config["tokens_out"] += get_token_count(
                    str(res.tool_name) + str(res.arguments)
                )

            return res

        return wrapper
    
    def append_conversation_history(self, previous_qa: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Append the conversation history in a structured format.
        
        Args:
            previous_qa: List of (user_message, assistant_message) tuples representing the conversation history.

        Returns:
            List[Dict[str, Any]]: The updated conversation history.
        """
        for user_msg, assistant_msg in previous_qa:
            self.context.append({"role": "user", "content": user_msg})
            self.context.append({"role": "assistant", "content": assistant_msg})
        return self.context

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent with its required configurations."""
        self.context = []

    @abstractmethod
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate a plain-text response.

        Args:
            user_message: The user's message.
            system_prompt: Optional system-role instruction injected as the first
                message in the conversation (replaces any previously stored system
                prompt when supplied).
            context: Optional explicit history to use instead of the stored one.
            temperature: Sampling temperature.

        Returns:
            str: The assistant's text reply.
        """
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: str = "any",
    ) -> ToolCallResult:
        """Ask the model to call one of the provided tools.

        Args:
            user_message: The user-facing question / instruction.
            tools: List of tool-schema dicts
                   (``{"type": "function", "function": {...}}``).
            system_prompt: Optional system instruction.
            tool_choice: "any" forces a tool call (mapped to "required" for
                         OpenAI-compatible providers).

        Returns:
            ToolCallResult
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
        tool_choice: str = "any",
    ) -> ToolCallResult:
        """
        Append a tool-role message and ask the model for its next tool call.
        Used to continue a tool-calling loop without breaking the conversation.

        Args:
            tool_call_id:     ID from the preceding generate_with_tools call.
            tool_name:        Name of the function that was executed.
            result:           Serialised result of executing the tool.
            next_instruction: User-turn message prompting the next action.
            tools:            Available tool schemas.
            tool_choice:      "any"/"required" both force a tool call.

        Returns:
            ToolCallResult
        """
        pass

    def follow_up_question(self, question: str) -> str:
        """Continue the conversation using the stored context."""
        return self.generate_response(question)

    def reset_context(self) -> None:
        """Clear the stored conversation history."""
        self.context = []

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for identification."""
        pass
