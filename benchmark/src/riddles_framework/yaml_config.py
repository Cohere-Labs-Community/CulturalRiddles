from __future__ import annotations

import yaml

from .constants import ResponseFormat, RiddleLanguagePresented
from .exceptions import ConfigError
from .schemas import ModelCapabilities, ModelSpec, RunConfig


def _opt_float(value):
    """Return float(value) if value is set, else None.

    Distinguishes 'not set' from 'set to 0.0' — important for temperature,
    where None means 'rely on model default' and 0.0 means 'really, zero'.
    """
    if value is None:
        return None
    return float(value)


def _opt_positive_int(value, *, key: str):
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ConfigError(f"{key} must be >= 1 when provided")
    return parsed


def _local_cache_client_mode(value):
    mode = value or "sync"
    if mode not in {"sync", "async"}:
        raise ConfigError("local_cache_client_mode must be 'sync' or 'async'")
    return mode


def load_run_config_from_yaml(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        model_data = data["model_spec"]
        caps_data = model_data.get("capabilities", {})

        spec = ModelSpec(
            model_name=model_data["model_name"],
            model_id=model_data["model_id"],
            provider=model_data["provider"],
            access_pattern=model_data["access_pattern"],
            provider_model_name=model_data["provider_model_name"],
            capabilities=ModelCapabilities(
                supports_reasoning=bool(caps_data.get("supports_reasoning", False)),
                supports_search=bool(caps_data.get("supports_search", False)),
                supports_structured_output=bool(caps_data.get("supports_structured_output", False)),
                supports_batch=bool(caps_data.get("supports_batch", False)),
                supports_seed=bool(caps_data.get("supports_seed", False)),
            ),
            size_label=model_data.get("size_label"),
            closed_source=model_data.get("closed_source"),
            notes=model_data.get("notes"),
        )

        model_params = data.get("model_params") or {}
        if not isinstance(model_params, dict):
            raise ConfigError("model_params must be a mapping if provided")

        run_batches = data.get("run_batches")
        if run_batches is not None:
            if not isinstance(run_batches, list) or not all(
                isinstance(b, str) for b in run_batches
            ):
                raise ConfigError("run_batches must be a list of strings if provided")

        return RunConfig(
            model_spec=spec,
            runner_name=data["runner_name"],
            dataset_path=data["dataset_path"],
            dataset_version=data["dataset_version"],
            prompt_template_id=data["prompt_template_id"],
            prompt_template_version=data["prompt_template_version"],
            prompt_language=data["prompt_language"],
            riddle_language_presented=RiddleLanguagePresented(data["riddle_language_presented"]),
            is_hint_provided=bool(data["is_hint_provided"]),
            reasoning_mode=data.get("reasoning_mode"),
            search_enabled=bool(data.get("search_enabled", False)),
            response_format=ResponseFormat(data.get("response_format", "free_form")),
            batch_mode=bool(data.get("batch_mode", False)),
            batch_size=_opt_positive_int(data.get("batch_size"), key="batch_size"),
            run_batches=run_batches,
            # Temperature: only set if the user wrote it; else rely on model default.
            temperature=_opt_float(data.get("temperature")),
            max_output_tokens=int(data.get("max_output_tokens", 256)),
            seed=data.get("seed"),
            model_params=dict(model_params),
            cache_enabled=bool(data.get("cache_enabled", False)),
            cache_backend=data.get("cache_backend"),
            cache_namespace=data.get("cache_namespace"),
            cache_dir=data.get("cache_dir"),
            local_cache_client_mode=_local_cache_client_mode(data.get("local_cache_client_mode")),
            api_key_env_var=data.get("api_key_env_var"),
            base_url=data.get("base_url"),
            runner_version=data.get("runner_version", "v1.0.0"),
            notes=data.get("notes"),
            strict_capability_check=bool(data.get("strict_capability_check", True)),
            strict_uploadable_rows=bool(data.get("strict_uploadable_rows", False)),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc}") from exc
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Failed to parse config YAML: {exc}") from exc
