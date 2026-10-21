from __future__ import annotations

from .registry import get_model_adapter_factory
from .schemas import ModelSpec


def adapter_key_for_spec(spec: ModelSpec) -> str:
    return f"{spec.provider}:{spec.access_pattern}"


def create_model(spec: ModelSpec, adapter_kwargs: dict | None = None):
    key = adapter_key_for_spec(spec)
    factory = get_model_adapter_factory(key)
    return factory(spec, adapter_kwargs or {})
