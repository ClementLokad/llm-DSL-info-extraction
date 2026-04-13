"""
ClaudeAgent – Anthropic Claude-powered agent with role-aware conversation + native tool-calling.

Conversation model
------------------
Every call builds and maintains ``self.context``, a list of Anthropic message dicts.
Note: Anthropic handles the system prompt separately (not as a message role).

The roles used in messages are:

  "user"      – the human turn
  "assistant" – the model's reply (text *or* tool_use blocks)
  "user"      – tool results are wrapped in user messages with tool_result content blocks

Plain-text flow  (generate_response)
  system  +  user  →  assistant  →  user  →  assistant  …

Tool-calling flow  (generate_with_tools + submit_tool_result_and_continue)
  system  +  user  →  assistant[tool_use]
               →  user[tool_result]  →  assistant[text]  …

Optimizations
-------------
• Prompt caching on system prompt (ephemeral cache control)
• Native tool-use support via Anthropic's tool schema
• Both Sonnet and Haiku model support with configurable selection
• Cost-aware token tracking using Anthropic's usage metadata
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from anthropic import Anthropic

from .base import LLMAgent, rate_limited, ToolCallResult
from config_manager import get_config


class ClaudeAgent(LLMAgent):
    """Anthropic Claude agent with role-aware history and native tool-calling."""

    def __init__(self, model: str = "claude-4-6-sonnet-latest"):
        """
        Initialize a Claude agent.

        Args:
            model: Claude model identifier
                   - "claude-3-5-sonnet-latest" (default, more capable)
                   - "claude-3-5-haiku-latest" (faster, cheaper)
        """
        super().__init__()
        self.client: Optional[Anthropic] = None
        self.api_key: Optional[str] = None
        self._model = model
        self.context: List[Dict[str, Any]] = []
        self._system_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load credentials and create Anthropic client.
        
        Note: We skip connectivity testing here to avoid consuming credits during
        initialization. Connectivity issues will be caught on first actual API call
        (generate_response, generate_with_tools, etc).
        """
        config = get_config()
        self.api_key = config.get_api_key("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        # Create client without testing connectivity (no credit consumption)
        self.client = Anthropic(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self.client or not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

    @staticmethod
    def _inject_system_prompt(system_prompt: str) -> str:
        """
        Return the system prompt (stored separately in Anthropic API).
        Helper method for consistency with other agents.
        """
        return system_prompt

    def _extract_text(self, response: Any) -> str:
        """Pull the assistant's text content from a chat completion response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _extract_tool_call(self, response: Any) -> ToolCallResult:
        """
        Pull the first tool_use block from an assistant message and return a
        provider-agnostic ToolCallResult.
        """
        for block in response.content:
            if block.type == "tool_use":
                # Anthropic's tool_use block has: id, name, input (already a dict)
                return ToolCallResult(
                    tool_name=block.name,
                    tool_id=block.id,
                    arguments=block.input,
                )
        raise ValueError(
            f"Claude did not return a tool call. "
            f"Raw response: {json.dumps([{'type': block.type,'content': getattr(block, 'text', getattr(block, 'input', None))} for block in response.content], ensure_ascii=False)}"
        )

    # ------------------------------------------------------------------
    # Public API – plain-text generation
    # ------------------------------------------------------------------

    @rate_limited()
    @LLMAgent.count_tokens
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a plain-text assistant reply.

        Stateless mode  – pass an explicit *context*; the instance's own
            history is neither read nor written.
        Stateful mode   – omit *context*; ``self.context`` is updated in place
            so subsequent calls build a multi-turn conversation.

        In both modes *system_prompt* is passed as Anthropic's system parameter
        (not as a message role) and may be updated independently for each call.

        Args:
            user_message: The user turn content.
            system_prompt: Optional static instruction for the model persona.
            context: Explicit history override (not stored back on self).
            temperature: Sampling temperature.

        Returns:
            str: The assistant's text reply.
        """
        self._ensure_initialized()
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")

        stateless = context is not None

        # Work on a local copy so we never accidentally mutate the caller's list
        history: List[Dict[str, Any]] = list(context if stateless else self.context)

        # In stateful mode, update stored system prompt and context
        if system_prompt:
            self._system_prompt = system_prompt
            if not stateless:
                self.context = history

        history.append({"role": "user", "content": user_message})
        if not stateless:
            self.context.append({"role": "user", "content": user_message})

        # Build the API call with system prompt as a separate parameter
        payload = {
            "model": self._model,
            "messages": history,
            "temperature": temperature,
            "max_tokens": 5000,
        }

        # Add system prompt with cache control for cost optimization
        if system_prompt or self._system_prompt:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system_prompt or self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = self.client.messages.create(**payload)
        reply = self._extract_text(response)

        if not stateless:
            self.context.append({"role": "assistant", "content": reply})

        return reply

    # ------------------------------------------------------------------
    # Public API – tool-calling
    # ------------------------------------------------------------------

    @rate_limited()
    @LLMAgent.count_tokens
    def generate_with_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: str = "any",
    ) -> ToolCallResult:
        """
        Ask the model to select and call one of the supplied tools.

        The assistant message (containing ``tool_use``) is appended to
        ``self.context`` so that ``submit_tool_result_and_continue`` can close the loop.

        Args:
            user_message: The user-visible prompt / question.
            tools: List of ``{"type": "function", "function": {...}}`` dicts
                   (OpenAI-compatible format; Anthropic SDK handles conversion).
            system_prompt: Optional persona / instruction for the model.
            tool_choice: ``"any"`` or ``"auto"`` (maps to required/auto in Anthropic).

        Returns:
            ToolCallResult
        """
        self._ensure_initialized()

        if system_prompt:
            self._system_prompt = system_prompt

        self.context.append({"role": "user", "content": user_message})

        # Convert OpenAI-style tools to Anthropic format if needed
        # Anthropic SDK accepts both formats, but we normalize for clarity
        anthropic_tools = self._normalize_tools(tools)

        payload = {
            "model": self._model,
            "messages": self.context,
            "tools": anthropic_tools,
            "temperature": 0.05,
            "max_tokens": 5000,
        }

        # Map tool_choice values: "any" → "required", "auto" → "auto"
        if tool_choice == "any":
            payload["tool_choice"] = {"type": "any"}

        # Add system prompt with cache control
        if self._system_prompt:
            payload["system"] = [
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = self.client.messages.create(**payload)
        tool_call = self._extract_tool_call(response)

        # Persist the raw assistant message so that the subsequent tool-role
        # message has a valid predecessor in history.
        # Convert response to a dict format for storage
        assistant_msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_call.tool_id,
                    "name": tool_call.tool_name,
                    "input": tool_call.arguments,
                }
            ],
        }
        self.context.append(assistant_msg)

        return tool_call

    @rate_limited()
    @LLMAgent.count_tokens
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
            tool_call_id: ID from the preceding generate_with_tools call.
            tool_name: Name of the function that was executed.
            result: Serialised result of executing the tool.
            next_instruction: User-turn message prompting the next action.
            tools: Available tool schemas.
            tool_choice: "any"/"auto" for tool forcing.

        Returns:
            ToolCallResult
        """
        self._ensure_initialized()

        # Append tool result in Anthropic format
        self.context.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            ],
        })

        # Append next instruction as a user message
        self.context.append({"role": "user", "content": next_instruction})

        anthropic_tools = self._normalize_tools(tools)

        payload = {
            "model": self._model,
            "messages": self.context,
            "tools": anthropic_tools,
            "temperature": 0.05,
            "max_tokens": 5000,
        }

        if tool_choice == "any":
            payload["tool_choice"] = {"type": "any"}

        if self._system_prompt:
            payload["system"] = [
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = self.client.messages.create(**payload)
        tool_call = self._extract_tool_call(response)

        # Persist assistant response
        assistant_msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_call.tool_id,
                    "name": tool_call.tool_name,
                    "input": tool_call.arguments,
                }
            ],
        }
        self.context.append(assistant_msg)

        return tool_call

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tool schema to Anthropic format if needed.

        Anthropic expects:
          [{"name": "...", "description": "...", "input_schema": {...}}]

        OpenAI style (which we accept):
          [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

        This method detects the format and converts if needed.
        """
        if not tools:
            return []

        # Check if already in Anthropic format (has 'name' key at top level)
        if "name" in tools[0]:
            return tools

        # Convert from OpenAI format
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                anthropic_tool = {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
                anthropic_tools.append(anthropic_tool)
            else:
                # Already in some other format, pass through
                anthropic_tools.append(tool)

        return anthropic_tools

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"Claude-{self._model}"


if __name__ == "__main__":
    agent = ClaudeAgent(model="claude-sonnet-4-6")
    agent.initialize()

    # Test plain-text generation
    response = agent.generate_response(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful assistant. Answer concisely.",
    )
    print(f"Response: {response}\n")

    # Test tool-calling (example)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            },
        }
    ]

    tool_call = agent.generate_with_tools(
        user_message="Add 5 and 3",
        tools=tools,
        system_prompt="You are a helpful math assistant.",
    )
    print(f"Tool call: {tool_call.tool_name}({tool_call.arguments})")
