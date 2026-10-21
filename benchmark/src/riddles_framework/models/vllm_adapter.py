from __future__ import annotations

from .openai_adapter import OpenAIAdapter


class VLLMAdapter(OpenAIAdapter):
    """Adapter for OpenAI-compatible model servers (vLLM, Ollama, LM Studio, ...).

    These servers expose the OpenAI chat-completions API, so the behaviour is
    identical to :class:`OpenAIAdapter`; only the config differs (point
    ``base_url`` at the server and set ``api_key_env_var`` to the token it
    expects, if any). Registered under ``vllm:direct_api`` in ``cli.py``.
    """
