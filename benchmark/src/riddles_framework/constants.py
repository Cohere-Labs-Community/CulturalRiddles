from __future__ import annotations

from enum import Enum


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    EMPTY = "empty"
    REFUSAL = "refusal"
    UNSUPPORTED_CONFIG = "unsupported_config"
    SKIPPED = "skipped"


class ResponseFormat(str, Enum):
    FREE_FORM = "free_form"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


class RiddleLanguagePresented(str, Enum):
    SOURCE = "source"
    ENGLISH = "english"


class AccessPattern(str, Enum):
    DIRECT_API = "direct_api"
    GCP = "gcp"
    OPENROUTER = "openrouter"
    HUGGINGFACE_ROUTER = "huggingface_router"
    HUGGINGFACE_TEXT_GENERATION = "huggingface_text_generation"
    TOGETHER = "together"
    LOCAL = "local"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    ALIBABA = "alibaba"
    AI2 = "ai2"
    MISTRAL = "mistral"
    XAI = "xai"
    QWEN = "qwen"
    OPENROUTER = "openrouter"
    HUGGINGFACE_ROUTER = "huggingface_router"
    HUGGINGFACE_TEXT_GENERATION = "huggingface_text_generation"
    TOGETHER = "together"
    LOCAL = "local"
    DUMMY = "dummy"
