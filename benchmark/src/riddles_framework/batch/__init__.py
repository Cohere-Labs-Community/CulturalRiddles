"""Fast cache-filling backends for the riddles framework.

Two ways to fill CacheSaver's on-disk cache much faster than the serial
runner, then let a normal ``cache_enabled`` run read everything as cache hits:

- ``threadpool``  — concurrent real-time calls (N at once) for any
  OpenAI-compatible provider (Cohere, OpenRouter, Together, ...).
- ``batch`` (planned) — provider Batch APIs for OpenAI / Gemini / Anthropic.

Both write entries CacheSaver reads back verbatim; see ``cache_compat``.
"""
