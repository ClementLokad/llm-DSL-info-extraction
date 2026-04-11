"""
DeepSeekAgent – OpenAI-compatible API with native tool-calling and extended thinking.

Conversation model
------------------
Every call builds and maintains ``self.context``, a list of OpenAI-compatible
message dicts. The roles used are:

  "system"    – static instructions / persona (first message, never repeated)
  "user"      – the human turn
  "assistant" – the model's reply (text *or* tool_calls)
  "tool"      – the result of executing a tool, matched to its call by id

Plain-text flow  (generate_response)
  system  →  user  →  assistant  →  user  →  assistant  …

Tool-calling flow  (generate_with_tools + submit_tool_result)
  system  →  user  →  assistant[tool_calls]
                   →  tool[result]  →  assistant[text]  …

DeepSeek-specific optimizations
--------------------------------
* Supports both standard chat models (deepseek-chat) and reasoning models (R1).
* For R1 models, extended thinking can be enabled to improve complex reasoning.
* Leverages proper temperature and top_p settings for optimal token efficiency.
* Uses stop sequences when appropriate to reduce token waste.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .base import LLMAgent, rate_limited, ToolCallResult
from config_manager import get_config


class DeepSeekAgent(LLMAgent):
    """DeepSeek AI agent with OpenAI-compatible API and extended thinking support."""

    def __init__(self, model: str = "deepseek-chat"):
        """
        Args:
            model: DeepSeek model to use. Options include:
                   - "deepseek-chat" (default, V3-based)
                   - "deepseek-reasoner" (R1-based with extended thinking)
        """
        super().__init__()
        self.base_url = "https://api.deepseek.com/v1"
        self.api_key: Optional[str] = None
        self._model = model
        self.context: List[Dict[str, Any]] = []
        self._is_reasoning_model = "reasoner" in model.lower() or "r1" in model.lower()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Validate credentials and test connectivity."""
        config = get_config()
        self.api_key = config.get_api_key("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._auth_headers(),
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to DeepSeek API: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _ensure_initialized(self) -> None:
        if not self.api_key:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to /chat/completions and return the parsed JSON."""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=120,  # Extended timeout for reasoning models
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _inject_system_prompt(
        history: List[Dict[str, Any]], system_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Return a *new* list with system_prompt as the first message.

        If the history already starts with a system message it is replaced;
        otherwise the new message is prepended. The original list is never
        mutated, making this safe for both stateful and stateless call paths.
        """
        system_msg = {"role": "system", "content": system_prompt}
        if history and history[0]["role"] == "system":
            return [system_msg] + history[1:]
        return [system_msg] + history

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """
        Pull the assistant's text content from a chat completion response.
        
        For reasoning models, includes thinking context if available.
        """
        choice = data["choices"][0]
        message = choice["message"]
        
        # For reasoning models, prepend thinking if present
        content_parts = []
        if "thinking" in message:
            # Optionally include thinking in output (can be disabled for cleaner responses)
            pass
        
        content = message.get("content") or ""
        return content

    def _extract_tool_call(self, data: Dict[str, Any]) -> ToolCallResult:
        """
        Pull the first tool_call from an assistant message and return a
        provider-agnostic ToolCallResult.
        """
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            raise ValueError(
                "DeepSeek did not return a tool call. "
                f"Raw message: {json.dumps(message, ensure_ascii=False)}"
            )
        call = tool_calls[0]
        raw_args = call["function"]["arguments"]
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tool call arguments are not valid JSON: {raw_args}"
            ) from exc

        return ToolCallResult(
            tool_name=call["function"]["name"],
            tool_id=call["id"],
            arguments=arguments,
        )

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        """
        Build an optimized payload for DeepSeek API.
        
        Applies model-specific optimizations:
        - For reasoning models: uses extended thinking for complex tasks
        - Temperature and top_p tuning for token efficiency
        - Proper tool_choice mapping for tool-calling
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 8000,  # Increased for reasoning models
            "temperature": temperature,
        }

        # Extended thinking for reasoning models (R1)
        if self._is_reasoning_model and enable_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 4000,  # Allocate thinking budget
            }

        # Add top_p for controlled diversity
        if temperature > 0.5:
            payload["top_p"] = 0.95
        else:
            payload["top_p"] = 0.7  # More deterministic for tool calls

        # Tool configuration
        if tools:
            payload["tools"] = tools
            # Map internal tool_choice to OpenAI standard
            if tool_choice and tool_choice.lower() == "any":
                payload["tool_choice"] = "required"
            elif tool_choice:
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = "auto"

        return payload

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

        In both modes *system_prompt* is injected as the first message without
        mutating the caller's data structure.

        Args:
            user_message: The user turn content.
            system_prompt: Optional static instruction for the model persona.
            context: Explicit history override (not stored back on self).
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            str: The assistant's text reply.
        """
        self._ensure_initialized()
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")

        stateless = context is not None

        # Work on a local copy so we never accidentally mutate the caller's list
        history: List[Dict[str, Any]] = list(context if stateless else self.context)

        if system_prompt:
            history = self._inject_system_prompt(history, system_prompt)
            # Keep self.context in sync when in stateful mode
            if not stateless:
                self.context = history

        history.append({"role": "user", "content": user_message})
        if not stateless:
            self.context.append({"role": "user", "content": user_message})

        # Enable thinking for complex queries (heuristic: long messages)
        enable_thinking = self._is_reasoning_model and len(user_message) > 200

        payload = self._build_payload(
            messages=history,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

        data = self._post_chat(payload)
        reply = self._extract_text(data)

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

        The assistant message (containing ``tool_calls``) is appended to
        ``self.context`` so that ``submit_tool_result`` can close the loop.

        Args:
            user_message: The user-visible prompt / question.
            tools: List of ``{"type": "function", "function": {...}}`` dicts.
            system_prompt: Optional persona / instruction for the model.
            tool_choice: ``"any"`` forces a tool call (recommended).
                         ``"auto"`` lets the model decide.

        Returns:
            ToolCallResult
        """
        self._ensure_initialized()

        if system_prompt:
            self.context = self._inject_system_prompt(self.context, system_prompt)

        self.context.append({"role": "user", "content": user_message})

        # For tool calling, use lower temperature for deterministic behavior
        # Enable thinking for complex tool selection tasks
        enable_thinking = self._is_reasoning_model and len(user_message) > 100

        payload = self._build_payload(
            messages=self.context,
            temperature=0.05,  # Very low for deterministic tool selection
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
        )

        data = self._post_chat(payload)
        tool_call = self._extract_tool_call(data)

        # Persist the raw assistant message (contains tool_calls) so that the
        # subsequent tool-role message has a valid predecessor in history.
        self.context.append(data["choices"][0]["message"])

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
        """
        self.context.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        self.context.append({"role": "user", "content": next_instruction})

        # Enable thinking if this is a complex multi-step task
        enable_thinking = self._is_reasoning_model and len(next_instruction) > 100

        payload = self._build_payload(
            messages=self.context,
            temperature=0.05,
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
        )

        data = self._post_chat(payload)
        tool_call = self._extract_tool_call(data)
        self.context.append(data["choices"][0]["message"])

        return tool_call

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"DeepSeek-{self._model}"


if __name__ == "__main__":
    agent = DeepSeekAgent(model="deepseek-chat")
    agent.initialize()
    response = agent.generate_response(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful assistant. Answer concisely and end all of your sentences with an exclamation mark.",
    )
    print(f"A: {response}")

    # Example with reasoning model
    # agent = DeepSeekAgent(model="deepseek-reasoner")
    # agent.initialize()
    # response = agent.generate_response(
    #     user_message="Solve this complex problem: ...",
    #     system_prompt="You are an expert problem solver.",
    # )
    # print(f"Reasoning A: {response}")
