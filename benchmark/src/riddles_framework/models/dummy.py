from __future__ import annotations

import time
from typing import Any

from ..constants import GenerationStatus
from ..schemas import GenerationResult, ModelCapabilities, ModelSpec
from .base import Model


class DummyModel(Model):
    def __init__(self, spec: ModelSpec | None = None) -> None:
        if spec is None:
            spec = ModelSpec(
                model_name="Dummy Echo",
                model_id="dummy/echo-v1",
                provider="dummy",
                access_pattern="local",
                provider_model_name="dummy-echo-v1",
                capabilities=ModelCapabilities(
                    supports_reasoning=False,
                    supports_search=False,
                    supports_structured_output=True,
                    supports_batch=False,
                    supports_seed=False,
                ),
                closed_source=False,
                notes="Dummy model for framework testing",
            )
        super().__init__(spec)

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
        if response_format in {"json", "json_schema"}:
            text = '{"riddle_answer": "dummy_answer"}'
        else:
            text = "dummy_answer"
        latency_ms = int((time.time() - start) * 1000)

        return GenerationResult(
            raw_response=text,
            status=GenerationStatus.SUCCESS,
            latency_ms=latency_ms,
            provider_details={
                "provider_model_name": self.provider_model_name,
                "temperature": temperature,
                "reasoning_mode": reasoning_mode,
                "search_enabled": search_enabled,
                "response_format": response_format,
                "model_params": dict(model_params or {}),
            },
        )
