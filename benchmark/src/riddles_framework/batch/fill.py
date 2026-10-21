"""Fill a run's CacheSaver cache fast, then run normally for instant cache hits.

Builds the same prompts the runner would (same config / template / dataset),
finds cache misses, and fetches them either concurrently (thread pool, for
OpenAI-compatible providers) or via a provider Batch API (OpenAI / Anthropic /
Gemini). Results are written into CacheSaver's on-disk cache, so a subsequent

    uv run riddles-framework --config CONFIG

Usage (from the repository root):
    uv run python -m riddles_framework.batch.fill \\
        --config benchmark/configs/benchmark_v0/cohere_tiny_aya.yaml
    uv run python -m riddles_framework.batch.fill \\
        --config benchmark/configs/benchmark_v0/gpt55_none.yaml --backend batch
    uv run python -m riddles_framework.batch.fill \\
        --config CONFIG --backend threadpool --workers 12
"""

from __future__ import annotations

import argparse
import os

from ..dataset_loader import load_benchmark_instances
from ..prompts import build_prompt, load_prompt_template
from ..schemas import PromptConfig
from ..yaml_config import load_run_config_from_yaml

# Providers reached over an OpenAI-compatible chat endpoint (use threadpool).
OPENAI_COMPATIBLE = {
    "cohere",
    "openrouter",
    "huggingface_router",
    "qwen",
    "xai",
    "together",
    "openai",
    "vllm",
}
# Providers with a native Batch API and a distinct cache entry format.
BATCH_PROVIDERS = {"openai", "google", "anthropic"}


def _openai_extra_body(rc) -> dict:
    """OpenAI request kwargs shared by batch fill and dry-run cache keys.

    Keep this in sync with models/openai_adapter.py so Batch API cache entries
    match the normal runner's CacheSaver keys.
    """
    extra = {}
    if rc.temperature is not None:
        extra["temperature"] = rc.temperature
    if rc.seed is not None:
        extra["seed"] = rc.seed
    if rc.reasoning_mode:
        extra["reasoning_effort"] = rc.reasoning_mode
    if rc.search_enabled:
        extra["web_search"] = True
    if rc.model_params:
        extra.update(rc.model_params)
    return extra


def _build_cache_key(prompt: str, rc, provider: str, model: str) -> str:
    """Canonical CacheSaver key for a prompt, mirroring the fill backends.

    Single source of truth so `--dry-run` miss counts match what the actual fill
    (openai_batch / threadpool / concurrent_batch / gemini_batch) would key on.
    OpenAI uses `max_completion_tokens`; other OpenAI-compatible providers use
    `max_tokens`; Gemini keys on `{model, contents}`.
    """
    from .cache_compat import gemini_request_key, openai_request_key

    if provider == "google":
        return gemini_request_key({"model": model, "contents": prompt})
    if provider == "huggingface_text_generation":
        from .huggingface_text import build_hf_text_body, hf_text_request_key

        return hf_text_request_key(
            build_hf_text_body(
                prompt,
                model=model,
                max_output_tokens=rc.max_output_tokens,
                response_format=rc.response_format.value,
                extra=dict(rc.model_params),
            )
        )
    tok = "max_completion_tokens" if provider == "openai" else "max_tokens"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        tok: rc.max_output_tokens,
    }
    if rc.response_format.value == "json":
        body["response_format"] = {"type": "json_object"}
    if provider == "openai":
        body.update(_openai_extra_body(rc))
    elif rc.model_params:
        body.update(rc.model_params)
    return openai_request_key(body)


def _build_prompts(rc) -> list[str]:
    template = load_prompt_template(rc.prompt_template_id, rc.prompt_template_version)
    instances = load_benchmark_instances(rc.dataset_path, run_batches=rc.run_batches)
    pc = PromptConfig(
        prompt_language=rc.prompt_language,
        riddle_language_presented=rc.riddle_language_presented,
        is_hint_provided=rc.is_hint_provided,
        response_format=rc.response_format,
        reasoning_mode=rc.reasoning_mode,
    )
    return [build_prompt(inst, template, pc) for inst in instances]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fill a run's CacheSaver cache concurrently or via Batch API"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--backend", choices=["auto", "threadpool", "batch"], default="auto")
    p.add_argument("--workers", type=int, default=8, help="threadpool concurrency")
    p.add_argument("--poll", type=int, default=30, help="batch poll interval (s)")
    p.add_argument("--dry-run", action="store_true", help="report misses/backend, make no calls")
    p.add_argument(
        "--run-batch",
        dest="run_batches",
        nargs="+",
        default=None,
        metavar="BATCH",
        help=(
            "Fill only these submission batches, e.g. --run-batch v1, or "
            "--run-batch v0 v1. Overrides run_batches in the config. "
            "Default: fill all batches in the dataset."
        ),
    )
    args = p.parse_args(argv)

    rc = load_run_config_from_yaml(args.config)
    if args.run_batches is not None:
        rc.run_batches = args.run_batches
    if rc.run_batches is not None:
        print(f"Run-batch filter: {rc.run_batches}")
    if not rc.cache_enabled or not rc.cache_dir:
        p.error("config must have cache_enabled: true and a cache_dir for cache filling")

    provider = rc.model_spec.provider
    model = rc.model_spec.provider_model_name
    cache_dir = rc.cache_dir
    rf = rc.response_format.value
    backend = args.backend
    if backend == "auto":
        backend = "batch" if provider in BATCH_PROVIDERS else "threadpool"

    prompts = _build_prompts(rc)
    print(
        f"[fill] {len(prompts)} prompts | provider={provider} model={model} "
        f"backend={backend} cache_dir={cache_dir}"
    )

    if args.dry_run:
        from .cache_compat import cache_lookup

        miss = sum(
            1
            for pr in prompts
            if not cache_lookup(cache_dir, _build_cache_key(pr, rc, provider, model))
        )
        print(f"[fill] dry-run: {miss} misses, {len(prompts) - miss} already cached")
        return 0

    api_key = os.environ.get(rc.api_key_env_var) if rc.api_key_env_var else None
    if rc.api_key_env_var and not api_key:
        p.error(f"{rc.api_key_env_var} is not set")

    if backend == "threadpool":
        if provider == "huggingface_text_generation":
            from .huggingface_text import fill_huggingface_text_generation

            res = fill_huggingface_text_generation(
                prompts=prompts,
                model=model,
                cache_dir=cache_dir,
                api_key=api_key,
                max_workers=args.workers,
                max_output_tokens=rc.max_output_tokens,
                response_format=rf,
                extra_create_kwargs=dict(rc.model_params),
                base_url=rc.base_url,
                progress=lambda d, t: print(f"\r[fill] {d}/{t}", end="", flush=True),
            )
            print(f"\n[fill] done: {res}")
            return 0 if res.errors == 0 else 1
        if provider not in OPENAI_COMPATIBLE:
            p.error(
                f"threadpool backend supports OpenAI-compatible providers only, not {provider!r}"
            )
        from .threadpool import fill_openai_compatible

        res = fill_openai_compatible(
            prompts=prompts,
            model=model,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=rc.base_url,
            max_workers=args.workers,
            max_output_tokens=rc.max_output_tokens,
            response_format=rf,
            extra_create_kwargs=dict(rc.model_params),
            progress=lambda d, t: print(f"\r[fill] {d}/{t}", end="", flush=True),
        )
        print(f"\n[fill] done: {res}")
        return 0 if res.errors == 0 else 1

    # backend == "batch"
    if provider == "openai":
        from .openai_batch import run_openai_batch

        out = run_openai_batch(
            prompts=prompts,
            model=model,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=rc.base_url,
            max_output_tokens=rc.max_output_tokens,
            response_format=rf,
            extra_body=_openai_extra_body(rc),
            poll_seconds=args.poll,
        )
    elif provider == "anthropic":
        from .anthropic_batch import run_anthropic_batch

        out = run_anthropic_batch(
            prompts=prompts,
            model=model,
            cache_dir=cache_dir,
            api_key=api_key,
            max_output_tokens=rc.max_output_tokens,
            poll_seconds=args.poll,
        )
    elif provider == "google":
        from .gemini_batch import run_gemini_batch

        out = run_gemini_batch(
            prompts=prompts,
            model=model,
            cache_dir=cache_dir,
            api_key=api_key,
            poll_seconds=args.poll,
        )
    elif provider == "vllm":
        from .vllm_batch import run_vllm_batch

        out = run_vllm_batch(
            prompts=prompts,
            model=model,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=rc.base_url,
            max_output_tokens=rc.max_output_tokens,
            response_format=rf,
            extra_body=_openai_extra_body(rc),
        )
    else:
        p.error(f"no Batch API backend for provider {provider!r}")
    print(f"[fill] done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
