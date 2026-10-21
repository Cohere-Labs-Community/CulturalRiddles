from __future__ import annotations

import time
from typing import Any

from ..batch.cache_compat import cache_lookup, cache_store
from ..batch.huggingface_text import (
    build_hf_text_body,
    generate_hf_text,
    hf_text_request_key,
)
from ..constants import GenerationStatus
from ..schemas import GenerationResult, ModelSpec
from ..utils import getenv_required
from .base import Model


class HuggingFaceTextGenerationAdapter(Model):
    def __init__(self, spec: ModelSpec, cfg: dict) -> None:
        super().__init__(spec)

        api_key_env_var = cfg.get("api_key_env_var", "HF_TOKEN")
        self._api_key = getenv_required(api_key_env_var)
        self._cache_dir = cfg.get("cache_dir") if cfg.get("cache_enabled", False) else None
        self._base_url = cfg.get("base_url")

    def _cache_key(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        response_format: str,
        model_params: dict[str, Any] | None,
    ) -> str:
        body = build_hf_text_body(
            prompt,
            model=self.provider_model_name,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            extra=model_params,
        )
        return hf_text_request_key(body)

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
        params = dict(model_params or {})
        if temperature is not None:
            params["temperature"] = temperature
        if seed is not None:
            params["seed"] = seed

        key = self._cache_key(
            prompt,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            model_params=params,
        )
        if self._cache_dir:
            cached = cache_lookup(self._cache_dir, key)
            if cached:
                return cached[0]

        try:
            result = generate_hf_text(
                prompt=prompt,
                model=self.provider_model_name,
                api_key=self._api_key,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                extra=params,
                base_url=self._base_url,
            )
            result.latency_ms = int((time.time() - start) * 1000)
            if self._cache_dir and result.status == GenerationStatus.SUCCESS:
                cache_store(self._cache_dir, key, result)
            return result
        except Exception as exc:
            return GenerationResult(
                raw_response=None,
                status=GenerationStatus.ERROR,
                error_message=str(exc),
                latency_ms=int((time.time() - start) * 1000),
            )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float | None = None,
        max_output_tokens: int = 256,
        seed: int | None = None,
        reasoning_mode: str | None = None,
        search_enabled: bool = False,
        response_format: str = "json",
        model_params: dict[str, Any] | None = None,
    ) -> list[GenerationResult]:
        return [
            self.generate(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
                reasoning_mode=reasoning_mode,
                search_enabled=search_enabled,
                response_format=response_format,
                model_params=model_params,
            )
            for prompt in prompts
        ]
