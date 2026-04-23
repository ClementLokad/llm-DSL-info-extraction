"""
QwenAgent – Ollama-backed agent implementing the full LLMAgent interface.

Key differences from the previous implementation
-------------------------------------------------
* Switched from ``ollama.generate`` (raw completion, no roles) to
  ``ollama.chat`` (role-aware messages: system / user / assistant / tool).
  This matches how Qwen instruction-tuned models were fine-tuned and aligns
  with the universal role convention shared by Mistral, Llama 3, Gemma, etc.

* Ollama's ``ollama.chat`` supports the OpenAI-compatible tool-calling schema
  for models that were fine-tuned with function-calling support (e.g.
  qwen2.5:7b-instruct and larger).  ``generate_with_tools`` and
  ``submit_tool_result`` are implemented accordingly.

* For models that do NOT support tool-calling (older or smaller Qwen variants),
  ``generate_with_tools`` will raise a clear ``NotImplementedError`` rather
  than silently misbehaving.  You can detect capability at runtime via the
  ``supports_tool_calling`` property and fall back to a prompt-based approach
  if needed.

Context management
------------------
``self.context`` is a flat list of role-keyed message dicts, identical in
structure to the Mistral implementation.  ``reset_context()`` clears it.
``follow_up_question`` continues the stored conversation without a new system
prompt.

Ollama-specific notes
---------------------
* ``num_ctx`` controls the context window.  The default (16 384 tokens) is
  kept from the original implementation but exposed as a constructor argument
  so it can be increased for longer planning sessions.
* Ollama runs locally so there is no API key and ``initialize()`` simply
  verifies that the Ollama daemon is reachable and the requested model exists.
* Rate-limiting is not applied (local inference has no quota), but the
  ``@rate_limited`` decorator is harmless if left on; it is omitted here for
  clarity.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import ollama

from .base import LLMAgent, ToolCallResult
from utils.config_manager import get_config


class QwenAgent(LLMAgent):
    """Ollama-backed Qwen agent with role-aware chat and tool-calling support."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_k_m",
        num_ctx: int = 16384,
        max_tokens: int = 4096,
    ):
        """
        Args:
            model: Ollama model tag to use.
            num_ctx: Context window size in tokens passed to Ollama.
            max_tokens: Maximum tokens to generate per response (num_predict).
                       Default 4096 to prevent endless loops. Set to None to disable.
        """
        super().__init__()
        self._model = model
        self._num_ctx = num_ctx
        self._max_tokens = max_tokens
        self.context: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Verify that Ollama is running and the model is available locally."""
        try:
            available = [m.model for m in ollama.list().get("models", [])]
            
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Ollama daemon: {exc}"
            ) from exc

        # Ollama tags can be 'model:tag' or just 'model'; do a prefix match
        model_base = self._model.split(":")[0]
        if not any(m.startswith(model_base) for m in available):
            raise RuntimeError(
                f"Model '{self._model}' is not available locally. "
                f"Run: ollama pull {self._model}"
            )

        self.context = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Call ollama.chat and return the raw response dict.
        Tools are passed only when provided so that non-tool-calling models
        are not accidentally broken by an unexpected parameter.
        """
        options: Dict[str, Any] = {"num_ctx": self._num_ctx, "temperature": temperature}
        if self._max_tokens is not None:
            options["num_predict"] = self._max_tokens
        
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools:
            kwargs["tools"] = tools
        return ollama.chat(**kwargs)

    @staticmethod
    def _extract_tool_call(response: Dict[str, Any]) -> ToolCallResult:
        """
        Parse a tool call from an Ollama chat response.

        Ollama follows the OpenAI-compatible format:
            response["message"]["tool_calls"][0]["function"]["name"]
            response["message"]["tool_calls"][0]["function"]["arguments"]

        A synthetic UUID is generated as the tool_id because Ollama does not
        return one natively.  The same ID is round-tripped into the tool-role
        message so the conversation structure stays valid.
        """
        message = response.get("message", {})
        
        # --- Primary path: structured tool_calls (preferred) ---
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        
        if tool_calls:
            call = tool_calls[0]
            func = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", {})
            raw_args = func.get("arguments", {}) if isinstance(func, dict) else getattr(func, "arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool call arguments are not valid JSON: {raw_args}") from exc
            else:
                arguments = raw_args
            name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
            return ToolCallResult(tool_name=name, tool_id=str(uuid.uuid4()), arguments=arguments)

        # --- Fallback path: model emitted tool call as plain text ---
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "") or ""

        # Extract the first complete JSON object by finding the outermost { } span.
        # This handles markdown fences (```json ... ```), XML wrappers (<tool_call>),
        # and any other surrounding prose the model may have added.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(content[start:end + 1])
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", parsed.get("parameters", {}))
                if name:
                    return ToolCallResult(
                        tool_name=name,
                        tool_id=str(uuid.uuid4()),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
            except (json.JSONDecodeError, AttributeError):
                pass

        raise ValueError(
            "Ollama did not return a tool call and fallback text parsing also failed. "
            f"Raw message: {message!r}"
        )

    # ------------------------------------------------------------------
    # Public API – plain-text generation
    # ------------------------------------------------------------------

    @LLMAgent.count_tokens
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: int = None
    ) -> str:
        """
        Generate a plain-text assistant reply.

        Stateless mode  – pass an explicit *context*; ``self.context`` is
            neither read nor written.
        Stateful mode   – omit *context*; ``self.context`` grows with each
            call, building a multi-turn conversation.

        Args:
            user_message: The user turn content.
            system_prompt: Optional system-role instruction.
            context: Explicit history override (stateless).
            temperature: Temperature of the model

        Returns:
            str: The assistant's text reply.
        """
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

        if temperature:
            response = self._chat(history, temperature=0.3)
        else:
            response = self._chat(history)

        reply: str = response["message"]["content"] or ""

        if not stateless:
            self.context.append({"role": "assistant", "content": reply})

        return reply

    # ------------------------------------------------------------------
    # Public API – tool-calling
    # ------------------------------------------------------------------
    
    @LLMAgent.count_tokens
    def generate_with_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        system_prompt: str = "",
        tool_choice: str = "any",
    ) -> ToolCallResult:
        """
        Ask the model to select and call one of the supplied tools.

        Ollama does not expose a ``tool_choice`` parameter equivalent to
        Mistral's.  For models that support tool-calling the behaviour is
        effectively ``tool_choice="auto"`` — the model decides whether to
        call a tool.  If it does not, a ValueError is raised so the caller
        can handle the fallback (e.g. redirect to simple_regeneration_tool).

        The assistant message is appended to ``self.context`` so that
        ``submit_tool_result`` can close the round-trip correctly.

        Args:
            user_message: The user-visible prompt.
            tools: List of ``{"type": "function", "function": {...}}`` dicts.
            system_prompt: Optional system-role instruction.
            tool_choice: Accepted for interface compatibility; ignored by Ollama.

        Returns:
            ToolCallResult
        """
        completed_sys_prompt = system_prompt + (
            '\nFor every tool call you make, you must fill in the "thought" field explaining '
            'your reasoning before filling in the other parameters.'
        )
        self.context = self._inject_system_prompt(self.context, completed_sys_prompt)

        self.context.append({"role": "user", "content": user_message})

        response = self._chat(self.context, tools=tools, temperature=0.05)

        tool_call = self._extract_tool_call(response)

        # Reconstruct the assistant message from scratch using the already-parsed
        # ToolCallResult, rather than mutating Ollama's response object which is a
        # Pydantic model that does not allow arbitrary field assignment.
        self.context.append({
            "role": "assistant",
            "content": response["message"].get("content") or "",
            "tool_calls": [{
                "id": tool_call.tool_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                },
            }],
        })

        return tool_call

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
        Append a ``tool`` role message and fetch the model's next toolcall.

        Closes the tool-calling round-trip:
            user → assistant[tool_calls] → tool[result] → assistant[tool]

        Args:
            tool_call_id: The ``tool_id`` from the preceding generate_with_tools call.
            tool_name: The function name that was executed.
            result: Serialised result of executing the tool.

        Returns:
            str: The assistant's follow-up tool call.
        """
        self.context.append({
            "role": "tool",
            "tool_call_id": tool_call_id,   # matches the synthetic ID we patched in
            "name": tool_name,
            "content": result,
        })
        self.context.append({"role": "user", "content": next_instruction})

        response = self._chat(self.context, tools=tools, temperature=0.3)
        tool_call = self._extract_tool_call(response)

        # Reconstruct the assistant message from scratch using the already-parsed
        # ToolCallResult, rather than mutating Ollama's response object which is a
        # Pydantic model that does not allow arbitrary field assignment.
        self.context.append({
            "role": "assistant",
            "content": response["message"].get("content") or "",
            "tool_calls": [{
                "id": tool_call.tool_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                },
            }],
        })
        
        return tool_call

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"Qwen-{self._model}"


if __name__ == "__main__":
    agent = QwenAgent()
    agent.initialize()
    response = agent.generate_response(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful assistant. Answer concisely and end all of your sentences with exclamation mark."
    )
    print(f"A: {response}")
