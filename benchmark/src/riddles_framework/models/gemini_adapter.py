from __future__ import annotations

import time
from typing import Any

from ..cache_clients import make_gemini_client
from ..constants import GenerationStatus
from ..schemas import GenerationResult, ModelSpec
from ..utils import getenv_required, looks_like_refusal
from .base import Model


class GeminiAdapter(Model):
    def __init__(self, spec: ModelSpec, cfg: dict) -> None:
        super().__init__(spec)

        api_key_env_var = cfg.get("api_key_env_var", "GEMINI_API_KEY")
        api_key = getenv_required(api_key_env_var)

        client_kwargs = {
            "api_key": api_key,
        }

        if cfg.get("cache_enabled", False):
            client_kwargs["namespace"] = cfg.get("cache_namespace")
            if cfg.get("cache_dir"):
                client_kwargs["cache_dir"] = cfg["cache_dir"]
            if cfg.get("batch_size"):
                client_kwargs["batch_size"] = cfg["batch_size"]
        self.client = make_gemini_client(cfg.get("cache_enabled", False), **client_kwargs)

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
        try:
            # google-genai's generate_content surface varies; this is best-effort.
            # The provider owner can refine without touching the core framework.
            call_kwargs: dict[str, Any] = {
                "model": self.provider_model_name,
                "contents": prompt,
            }
            if model_params:
                # Most genai parameters live under a `config=` argument; pass through
                # as `config` if the user supplied it explicitly, otherwise as top-level.
                if "config" in model_params:
                    call_kwargs["config"] = model_params["config"]
                else:
                    call_kwargs.update(model_params)

            resp = self.client.models.generate_content(**call_kwargs)
            text = getattr(resp, "text", None)
            latency_ms = int((time.time() - start) * 1000)

            status = GenerationStatus.SUCCESS
            if text is None or str(text).strip() == "":
                status = GenerationStatus.EMPTY
            elif looks_like_refusal(text):
                status = GenerationStatus.REFUSAL

            return GenerationResult(
                raw_response=text,
                status=status,
                latency_ms=latency_ms,
                provider_details={"model": self.provider_model_name},
            )
        except Exception as exc:
            return GenerationResult(
                raw_response=None,
                status=GenerationStatus.ERROR,
                error_message=str(exc),
                latency_ms=int((time.time() - start) * 1000),
            )
