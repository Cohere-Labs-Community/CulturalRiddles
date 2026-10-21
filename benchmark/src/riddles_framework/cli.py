from __future__ import annotations

import argparse

from .factory import adapter_key_for_spec, create_model
from .models.anthropic_adapter import AnthropicAdapter
from .models.cohere_adapter import CohereAdapter
from .models.dummy import DummyModel
from .models.gemini_adapter import GeminiAdapter
from .models.huggingface_router_adapter import HuggingFaceRouterAdapter
from .models.huggingface_text_generation_adapter import HuggingFaceTextGenerationAdapter

# Import adapters so registration is available.
from .models.openai_adapter import OpenAIAdapter
from .models.openrouter_adapter import OpenRouterAdapter
from .models.together_adapter import TogetherAdapter
from .models.transformers_local_adapter import LocalTransformersAdapter
from .models.vllm_adapter import VLLMAdapter
from .prompts import load_prompt_template
from .registry import list_registered_adapters, register_model_adapter
from .runner import BenchmarkRunner
from .yaml_config import load_run_config_from_yaml


def register_builtin_adapters() -> None:
    # Ignore duplicate registration on repeated CLI usage in same process.
    for key, factory in [
        ("dummy:local", lambda spec, cfg: DummyModel(spec)),
        ("openai:direct_api", lambda spec, cfg: OpenAIAdapter(spec, cfg)),
        ("anthropic:direct_api", lambda spec, cfg: AnthropicAdapter(spec, cfg)),
        ("google:direct_api", lambda spec, cfg: GeminiAdapter(spec, cfg)),
        ("cohere:direct_api", lambda spec, cfg: CohereAdapter(spec, cfg)),
        ("qwen:openrouter", lambda spec, cfg: OpenRouterAdapter(spec, cfg)),
        ("xai:openrouter", lambda spec, cfg: OpenRouterAdapter(spec, cfg)),
        ("openrouter:openrouter", lambda spec, cfg: OpenRouterAdapter(spec, cfg)),
        (
            "huggingface_router:huggingface_router",
            lambda spec, cfg: HuggingFaceRouterAdapter(spec, cfg),
        ),
        (
            "huggingface_text_generation:huggingface_text_generation",
            lambda spec, cfg: HuggingFaceTextGenerationAdapter(spec, cfg),
        ),
        ("together:together", lambda spec, cfg: TogetherAdapter(spec, cfg)),
        ("local:local", lambda spec, cfg: LocalTransformersAdapter(spec, cfg)),
        ("vllm:direct_api", lambda spec, cfg: VLLMAdapter(spec, cfg)),
    ]:
        try:
            register_model_adapter(key, factory)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Multicultural Riddles Benchmark Framework")
    parser.add_argument("--config", required=True, help="Path to YAML run config")
    parser.add_argument("--output-root", default="data/runs", help="Output directory for runs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the run to the first N dataset rows (for example, an offline dummy smoke)",
    )
    parser.add_argument(
        "--run-batch",
        dest="run_batches",
        nargs="+",
        default=None,
        metavar="BATCH",
        help=(
            "Run only these submission batches, e.g. --run-batch v1, or "
            "--run-batch v0 v1. Overrides run_batches in the config. "
            "Default: run all batches in the dataset."
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    register_builtin_adapters()

    run_config = load_run_config_from_yaml(args.config)
    adapter_key = adapter_key_for_spec(run_config.model_spec)
    if adapter_key not in list_registered_adapters():
        parser.error(
            f"Unsupported adapter {adapter_key!r}. Historical request metadata "
            "is evidence only; select a registered adapter for a new run."
        )
    if args.run_batches is not None:
        run_config.run_batches = args.run_batches
    if run_config.run_batches is not None:
        print(f"Run-batch filter: {run_config.run_batches}")

    prompt_template = load_prompt_template(
        run_config.prompt_template_id,
        run_config.prompt_template_version,
    )

    adapter_cfg = {
        "cache_enabled": run_config.cache_enabled,
        "cache_namespace": run_config.cache_namespace,
        "cache_dir": run_config.cache_dir,
        "cache_backend": run_config.cache_backend,
        "local_cache_client_mode": run_config.local_cache_client_mode,
        "api_key_env_var": run_config.api_key_env_var,
        "base_url": run_config.base_url,
        "model_params": run_config.model_params,
        "batch_size": run_config.batch_size,
    }

    if run_config.model_spec.provider == "dummy":
        model = DummyModel(run_config.model_spec)
    else:
        model = create_model(run_config.model_spec, adapter_cfg)

    runner = BenchmarkRunner(
        run_config=run_config,
        prompt_template=prompt_template,
        model=model,
        output_root=args.output_root,
        max_instances=args.limit,
    )
    run_id = runner.run()
    print(f"Run complete: {run_id}")


if __name__ == "__main__":
    main()
