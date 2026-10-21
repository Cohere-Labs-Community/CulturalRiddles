from __future__ import annotations

from typing import Callable

from .exceptions import ModelRegistrationError
from .schemas import ModelSpec

_MODEL_REGISTRY: dict[str, Callable[[ModelSpec, dict], object]] = {}


def register_model_adapter(key: str, factory: Callable[[ModelSpec, dict], object]) -> None:
    if key in _MODEL_REGISTRY:
        raise ModelRegistrationError(f"Adapter already registered for key={key}")
    _MODEL_REGISTRY[key] = factory


def get_model_adapter_factory(key: str) -> Callable[[ModelSpec, dict], object]:
    if key not in _MODEL_REGISTRY:
        raise ModelRegistrationError(f"No adapter registered for key={key}")
    return _MODEL_REGISTRY[key]


def list_registered_adapters() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())
