from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import GenerationResult, ModelSpec


class Model(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model_name = spec.model_name
        self.model_id = spec.model_id
        self.model_provider = spec.provider
        self.provider_model_name = spec.provider_model_name
        self.access_pattern = spec.access_pattern
        self.capabilities = spec.capabilities

    @abstractmethod
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
        raise NotImplementedError

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
                p,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
                reasoning_mode=reasoning_mode,
                search_enabled=search_enabled,
                response_format=response_format,
                model_params=model_params,
            )
            for p in prompts
        ]
