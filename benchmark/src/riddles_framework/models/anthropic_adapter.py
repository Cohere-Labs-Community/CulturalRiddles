from __future__ import annotations

import time
from typing import Any

from ..cache_clients import make_anthropic_client
from ..constants import GenerationStatus
from ..schemas import GenerationResult, ModelSpec
from ..utils import getenv_required, looks_like_refusal, safe_int
from .base import Model


class AnthropicAdapter(Model):
    def __init__(self, spec: ModelSpec, cfg: dict) -> None:
        super().__init__(spec)

        api_key_env_var = cfg.get("api_key_env_var", "ANTHROPIC_API_KEY")
        api_key = getenv_required(api_key_env_var)

        client_kwargs = {
            "api_key": api_key,
        }

        if cfg.get("base_url"):
            client_kwargs["base_url"] = cfg["base_url"]

        if cfg.get("cache_enabled", False):
            client_kwargs["namespace"] = cfg.get("cache_namespace")
            if cfg.get("cache_dir"):
                client_kwargs["cache_dir"] = cfg["cache_dir"]
            if cfg.get("batch_size"):
                client_kwargs["batch_size"] = cfg["batch_size"]
        self.client = make_anthropic_client(cfg.get("cache_enabled", False), **client_kwargs)
        # Optional override for the thinking budget. Defaults to max_output_tokens // 2 at call time.
        self.thinking_budget_tokens = cfg.get("thinking_budget_tokens")

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int = 256,
        seed: int | None = None,
        reasoning_mode: str | None = None,
        search_enabled: bool = False,
        response_format: str = "json",
        model_params: dict[str, Any] | None = None,
    ) -> GenerationResult:
        start = time.time()
        budget_tokens = None
        try:
            kwargs = {
                "model": self.provider_model_name,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Anthropic extended thinking requires temperature=1, and the
            # thinking budget must leave room for visible answer tokens.
            # UPDATE: extended reasoning no longer supported, now adaptive high effort thinking;
            # no longer take temperature and budget parameters
            if reasoning_mode:
                #    budget_tokens = self.thinking_budget_tokens
                #    if budget_tokens is None:
                #        budget_tokens = max(1, max_output_tokens // 2)

                kwargs["thinking"] = {
                    #        "type": "enabled",
                    "type": "adaptive",
                    #        "budget_tokens": budget_tokens,
                }
                kwargs["output_config"] = {
                    "effort": "high",
                }
                #    kwargs["temperature"] = 1
                kwargs.pop("temperature", None)
            # UPDATE: added to call web search tool, tool call needs to be forced or else
            # model tends to not use the search tool at all
            elif search_enabled:
                kwargs["tools"] = [
                    {
                        "type": "web_search_20260209",
                        "name": "web_search",
                    }
                ]
                kwargs["tool_choice"] = {"type": "any"}  # forces at least one tool cal
                if temperature is not None:
                    kwargs["temperature"] = temperature
            elif temperature is not None:
                kwargs["temperature"] = temperature
            # Merge arbitrary user-supplied model params LAST.
            if model_params:
                kwargs.update(model_params)

            resp = self.client.messages.create(**kwargs)

            text_parts = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))

            text = "\n".join(t for t in text_parts if t).strip() or None
            usage = getattr(resp, "usage", None)
            latency_ms = int((time.time() - start) * 1000)

            status = GenerationStatus.SUCCESS
            if text is None:
                status = GenerationStatus.EMPTY
            elif looks_like_refusal(text):
                status = GenerationStatus.REFUSAL

            return GenerationResult(
                raw_response=text,
                status=status,
                input_tokens=safe_int(getattr(usage, "input_tokens", None)) if usage else None,
                output_tokens=safe_int(getattr(usage, "output_tokens", None)) if usage else None,
                latency_ms=latency_ms,
                provider_details={
                    "id": getattr(resp, "id", None),
                    "model": getattr(resp, "model", None),
                    "stop_reason": getattr(resp, "stop_reason", None),
                    "thinking_enabled": bool(reasoning_mode),
                    "thinking_budget_tokens": budget_tokens if reasoning_mode else None,
                    "search_enabled": bool(search_enabled),
                    "search_requests": getattr(
                        getattr(usage, "server_tool_use", None), "web_search_requests", None
                    ),
                },
            )
        except Exception as exc:
            return GenerationResult(
                raw_response=None,
                status=GenerationStatus.ERROR,
                error_message=str(exc),
                latency_ms=int((time.time() - start) * 1000),
            )
