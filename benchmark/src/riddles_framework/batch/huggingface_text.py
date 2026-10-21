from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from ..constants import GenerationStatus
from ..schemas import GenerationResult
from ..utils import safe_int
from .cache_compat import cache_lookup, cache_store, openai_request_key


@dataclass
class FillResult:
    total: int
    hits: int
    filled: int
    errors: int


HF_CLIENT_KEYS = {"hf_provider", "hf_timeout", "hf_bill_to"}


def split_hf_kwargs(extra: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    source = dict(extra or {})
    client_kwargs = {
        "provider": source.pop("hf_provider", "auto"),
        "timeout": source.pop("hf_timeout", None),
        "bill_to": source.pop("hf_bill_to", None),
    }
    client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
    generation_kwargs = {k: v for k, v in source.items() if k not in HF_CLIENT_KEYS}
    return client_kwargs, generation_kwargs


def build_hf_text_body(
    prompt: str,
    *,
    model: str,
    max_output_tokens: int,
    response_format: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client_kwargs, generation_kwargs = split_hf_kwargs(extra)
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_new_tokens": max_output_tokens,
        "details": True,
        "return_full_text": False,
        "hf_provider": client_kwargs.get("provider", "auto"),
    }
    if response_format == "json":
        body["grammar"] = {"type": "json"}
    body.update(generation_kwargs)
    return body


def hf_text_request_key(body: dict[str, Any]) -> str:
    return openai_request_key(body)


def result_from_hf_text_output(output: Any) -> GenerationResult:
    text = getattr(output, "generated_text", None)
    if text is None:
        text = str(output) if output is not None else None

    details = getattr(output, "details", None)
    status = GenerationStatus.SUCCESS
    if text is None or str(text).strip() == "":
        status = GenerationStatus.EMPTY

    return GenerationResult(
        raw_response=text,
        status=status,
        input_tokens=None,
        output_tokens=safe_int(getattr(details, "generated_tokens", None)) if details else None,
        provider_details={
            "finish_reason": getattr(details, "finish_reason", None) if details else None,
            "seed": getattr(details, "seed", None) if details else None,
        },
    )


def _create_client(api_key: str, extra: dict[str, Any] | None, client_factory=None):
    client_kwargs, _ = split_hf_kwargs(extra)
    if client_factory is not None:
        return client_factory(token=api_key, **client_kwargs)
    from huggingface_hub import InferenceClient

    return InferenceClient(token=api_key, **client_kwargs)


def _create_client_with_base_url(
    api_key: str,
    extra: dict[str, Any] | None,
    base_url: str | None,
    client_factory=None,
):
    client_kwargs, _ = split_hf_kwargs(extra)
    if base_url:
        client_kwargs["base_url"] = base_url
        client_kwargs.pop("provider", None)
    if client_factory is not None:
        return client_factory(token=api_key, **client_kwargs)
    from huggingface_hub import InferenceClient

    return InferenceClient(token=api_key, **client_kwargs)


def generate_hf_text(
    *,
    prompt: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    response_format: str,
    extra: dict[str, Any] | None = None,
    base_url: str | None = None,
    client: Any | None = None,
) -> GenerationResult:
    if client is None:
        client = _create_client_with_base_url(api_key, extra, base_url)

    body = build_hf_text_body(
        prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        response_format=response_format,
        extra=extra,
    )
    _, generation_kwargs = split_hf_kwargs(extra)
    kwargs = {
        "model": model,
        "details": True,
        "return_full_text": False,
        "max_new_tokens": max_output_tokens,
    }
    if response_format == "json":
        kwargs["grammar"] = {"type": "json"}
    kwargs.update(generation_kwargs)

    output = client.text_generation(prompt, **kwargs)
    result = result_from_hf_text_output(output)
    result.provider_details = {
        **(result.provider_details or {}),
        "model": model,
        "hf_provider": body.get("hf_provider"),
    }
    return result


def fill_huggingface_text_generation(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str,
    max_workers: int = 8,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_create_kwargs: dict[str, Any] | None = None,
    base_url: str | None = None,
    max_attempts: int = 3,
    client_factory: Callable[..., Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FillResult:
    client = _create_client_with_base_url(api_key, extra_create_kwargs, base_url, client_factory)

    misses: list[tuple[str, dict[str, Any], str]] = []
    hits = 0
    for prompt in prompts:
        body = build_hf_text_body(
            prompt,
            model=model,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            extra=extra_create_kwargs,
        )
        key = hf_text_request_key(body)
        if cache_lookup(cache_dir, key):
            hits += 1
        else:
            misses.append((prompt, body, key))

    filled = errors = done = 0

    def work(item: tuple[str, dict[str, Any], str]) -> bool:
        prompt, _body, key = item
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = generate_hf_text(
                    prompt=prompt,
                    model=model,
                    api_key=api_key,
                    max_output_tokens=max_output_tokens,
                    response_format=response_format,
                    extra=extra_create_kwargs,
                    base_url=base_url,
                    client=client,
                )
                if (
                    result.status == GenerationStatus.SUCCESS
                    and str(result.raw_response or "").strip()
                ):
                    cache_store(cache_dir, key, result)
                    return True
                last_error = RuntimeError("provider returned an empty generation")
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"cache fill failed after {max_attempts} attempts: {last_error}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, m) for m in misses]
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
                filled += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors <= 5:
                    print(f"[fill] error: {exc}", flush=True)
            if progress:
                progress(done, len(misses))

    return FillResult(total=len(prompts), hits=hits, filled=filled, errors=errors)
