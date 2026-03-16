"""
GroqAgent – Groq-backed agent implementing the full LLMAgent interface.

Groq's API is OpenAI-compatible: same /chat/completions endpoint structure,
same message role conventions (system / user / assistant / tool), and the
same tool-calling schema ({"type": "function", "function": {...}}).

It therefore mirrors MistralAgent almost exactly. The only differences are:
  - Uses the groq SDK client instead of raw requests.
  - No dedicated connectivity-test endpoint — initialize() validates the key
    by listing available models.
  - tool_choice="required" is used instead of "any" (Groq follows the OpenAI
    spec where "required" forces a tool call; "any" is Mistral-specific).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from groq import Groq

from .base import LLMAgent, ToolCallResult, rate_limited
from config_manager import get_config


class GroqAgent(LLMAgent):
    """Groq-backed agent with role-aware history and native tool-calling."""

    def __init__(self, model: str = "qwen/qwen3-32b"):
        super().__init__()
        self.client: Optional[Groq] = None
        self.api_key: Optional[str] = None
        self._model = model
        self.context: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Validate credentials and test connectivity."""
        config = get_config()
        self.api_key = config.get_api_key("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        try:
            self.client = Groq(api_key=self.api_key)
            self.client.models.list()
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to Groq API: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

    @staticmethod
    def _inject_system_prompt(
        history: List[Dict[str, Any]], system_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Return a new list with system_prompt as the first message.
        Never mutates the original list.
        """
        system_msg = {"role": "system", "content": system_prompt}
        if history and history[0]["role"] == "system":
            return [system_msg] + history[1:]
        return [system_msg] + history
    
    def _extract_text(self, response: Any) -> str:
        content = response.choices[0].message.content or ""
        # Remove thinking blocks (empty or not) left in the output string.
        # Groq includes the tags even when thinking is disabled.
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        return content.strip()

    def _post_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Any:
        """Call the Groq chat completions endpoint and return the response."""
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 5000,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def _extract_tool_call(self, response: Any) -> ToolCallResult:
        """
        Pull the first tool_call from an assistant message and return a
        provider-agnostic ToolCallResult.
        """
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            raise ValueError(
                "Groq did not return a tool call. "
                f"Raw message: {message}"
            )
        call = tool_calls[0]
        raw_args = call.function.arguments
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tool call arguments are not valid JSON: {raw_args}"
            ) from exc

        return ToolCallResult(
            tool_name=call.function.name,
            tool_id=call.id,
            arguments=arguments,
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

        Stateless mode  – pass an explicit *context*; self.context is
            neither read nor written.
        Stateful mode   – omit *context*; self.context grows with each
            call, building a multi-turn conversation.

        Args:
            user_message:  The user turn content.
            system_prompt: Optional system-role instruction.
            context:       Explicit history override (stateless).
            temperature:   Sampling temperature.

        Returns:
            str: The assistant's text reply.
        """
        self._ensure_initialized()
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")

        stateless = context is not None
        history: List[Dict[str, Any]] = list(context if stateless else self.context)

        if system_prompt:
            history = self._inject_system_prompt(history, system_prompt)
            if not stateless:
                self.context = history

        history.append({"role": "user", "content": user_message})
        if not stateless:
            self.context.append({"role": "user", "content": user_message})

        response = self._post_chat(history, temperature=temperature)
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

        Note: Groq follows the OpenAI spec where the equivalent of Mistral's
        "any" is "required". The tool_choice parameter is accepted for
        interface compatibility and silently mapped: "any" → "required".

        Args:
            user_message:  The user-visible prompt.
            tools:         List of {"type": "function", "function": {...}} dicts.
            system_prompt: Optional system-role instruction.
            tool_choice:   "any" or "required" both force a tool call.

        Returns:
            ToolCallResult
        """
        self._ensure_initialized()

        # Map Mistral's "any" to OpenAI/Groq's "required"
        groq_tool_choice = "required" if tool_choice in ("any", "required") else "auto"

        if system_prompt:
            self.context = self._inject_system_prompt(self.context, system_prompt)

        self.context.append({"role": "user", "content": user_message})

        response = self._post_chat(
            self.context,
            tools=tools,
            tool_choice=groq_tool_choice,
            temperature=0.0,
        )
        tool_call = self._extract_tool_call(response)

        # Persist the assistant message so the subsequent tool-role message
        # has a valid predecessor in history.
        assistant_msg = response.choices[0].message
        self.context.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (assistant_msg.tool_calls or [])
            ],
        })

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

        Continues the tool-calling loop without breaking the conversation:
            assistant[tool_calls] → tool[result] → user[instruction]
            → assistant[tool_calls]

        Args:
            tool_call_id:     The tool_id from the preceding call.
            tool_name:        The function name that was executed.
            result:           Serialised result of executing the tool.
            next_instruction: User-turn message prompting the next action.
            tools:            Available tool schemas.
            tool_choice:      "any"/"required" both force a tool call.

        Returns:
            ToolCallResult
        """
        self._ensure_initialized()

        groq_tool_choice = "required" if tool_choice in ("any", "required") else "auto"

        self.context.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        self.context.append({"role": "user", "content": next_instruction})

        response = self._post_chat(
            self.context,
            tools=tools,
            tool_choice=groq_tool_choice,
            temperature=0.0,
        )
        tool_call = self._extract_tool_call(response)

        assistant_msg = response.choices[0].message
        self.context.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (assistant_msg.tool_calls or [])
            ],
        })

        return tool_call

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"Groq-{self._model}"

if __name__ == "__main__":
    agent = GroqAgent()
    agent.initialize()
    agent.reset_context()
    response = agent.generate_response(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful assistant. Answer concisely and end all of your sentences with exclamation mark."
    )
    print(f"A: {response}")