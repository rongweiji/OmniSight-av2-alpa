"""
Alpamayo-R1-10B inference client.

Wraps the vLLM OpenAI-compatible API for easy use.

Usage:
    client = AlpamayoClient(base_url="http://localhost:8000/v1")
    response = client.explain(system_prompt, user_prompt)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("openai package required: pip install openai") from e


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "alpamayo"


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    stop: list[str] = field(default_factory=list)


class AlpamayoClient:
    """
    Client for the Alpamayo-R1-10B vLLM server.

    Args:
        base_url:    URL of the running vLLM server (default: http://localhost:8000/v1)
        model_name:  Served model name as set in server.py (default: alpamayo)
        api_key:     API key if auth is enabled (default: reads ALPAMAYO_API_KEY env var)
        config:      Generation hyperparameters
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: str | None = None,
        config: GenerationConfig | None = None,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.config = config or GenerationConfig()

        resolved_key = api_key or os.getenv("ALPAMAYO_API_KEY", "not-required")
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def explain(
        self,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """
        Send a prompt to the model and return the text response.

        Args:
            system_prompt: Sets the model's role and behaviour.
            user_prompt:   The actual question / task content.
            config:        Override default generation config for this call.

        Returns:
            Model response as a plain string.
        """
        cfg = config or self.config
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    def explain_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig | None = None,
    ):
        """
        Stream the model response token by token.

        Yields:
            str chunks as they arrive from the server.

        Example:
            for chunk in client.explain_stream(sys_p, user_p):
                print(chunk, end="", flush=True)
        """
        cfg = config or self.config
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the server is reachable and the model is loaded."""
        try:
            models = self._client.models.list()
            return any(m.id == self.model_name for m in models.data)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"AlpamayoClient(base_url={self.base_url!r}, model={self.model_name!r})"
