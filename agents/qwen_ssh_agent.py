from typing import Any, Dict, List, Optional

from ollama import Client

from .qwen_agent import QwenAgent


class QwenSSHAgent(QwenAgent):
    """
    QwenAgent variant that routes all Ollama calls through a remote host
    (typically exposed via an SSH tunnel) instead of the local daemon.

    Only ``_chat`` and ``initialize`` are overridden — every other method
    (generate_response, generate_with_tools, submit_tool_result_and_continue,
    _extract_tool_call, …) is inherited unchanged from QwenAgent.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b-instruct",
        host: str = "http://localhost:11436",
        num_ctx: int = 16384,
    ):
        """
        Args:
            model:   Ollama model tag to use on the remote host.
            host:    URL of the remote Ollama server (default maps to a
                     typical SSH tunnel port).
            num_ctx: Context window size in tokens.
        """
        super().__init__(model=model, num_ctx=num_ctx)
        self._host = host
        self.client = Client(host=self._host)

    # ------------------------------------------------------------------
    # Initialisation  — same logic as QwenAgent but uses self.client
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Verify that the remote Ollama server is reachable and the model exists."""
        try:
            available = [m.model for m in self.client.list().get("models", [])]
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to remote Ollama server at {self._host}: {exc}"
            ) from exc

        model_base = self._model.split(":")[0]
        if not any(m.startswith(model_base) for m in available):
            raise RuntimeError(
                f"Model '{self._model}' is not available on {self._host}. "
                f"Run: ollama pull {self._model}  (on the remote host)"
            )

        self.context = []

    # ------------------------------------------------------------------
    # _chat  — only override needed: swap module-level ollama.chat for
    #          self.client.chat so calls go to the remote host
    # ------------------------------------------------------------------

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self._num_ctx, "temperature": temperature},
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.chat(**kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"QwenSSH-{self._model}@{self._host}"


if __name__ == "__main__":
    agent = QwenSSHAgent(host="http://localhost:11436")
    agent.initialize()
    response = agent.generate_response(
        user_message="What is the capital of Italy?",
        system_prompt="You are a helpful assistant. Answer concisely.",
    )
    print(f"A: {response}")