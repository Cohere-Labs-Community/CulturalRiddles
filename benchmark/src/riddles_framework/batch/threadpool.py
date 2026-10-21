"""Concurrent cache-fill for OpenAI-compatible providers (Cohere, OpenRouter,
Together, ...).

Runs up to ``max_workers`` real-time chat-completit requests at once against the
plain (thread-safe) OpenAI SDK client, writing each result into CacheSaver's
on-disk cache via ``cache_compat``. A subsequent ``cache_enabled`` run then
reads every prompt as a cache hit, so the slow part happens once, concurrently.

This deliberately bypasses CacheSaver's ``SyncWrapper`` (serial, asyncio-bound,
not thread-safe). Only cache *misses* are sent; existing entries are left alone.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from .cache_compat import cache_lookup, cache_store, request_key


@dataclass
class FillResult:
    total: int
    hits: int  # already cached, skipped
    filled: int  # newly fetched + cached
    errors: int


def _create_kwargs(
    prompt: str,
    model: str,
    max_output_tokens: int,
    response_format: str,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Match what the OpenAI-compatible adapters send to ``create``.

    Mirrors ``cohere_adapter`` / ``openrouter_adapter``: ``model`` / ``messages``
    / ``max_tokens`` (+ ``response_format`` only when JSON). Must stay identical
    so the reconstructed key matches CacheSaver's.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}
    if extra:
        kwargs.update(extra)
    return kwargs


def _openai_sdk_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Move OpenRouter-only body fields under OpenAI SDK's ``extra_body``."""
    sdk_kwargs = dict(kwargs)
    extra_body = dict(sdk_kwargs.pop("extra_body", {}) or {})
    if "reasoning" in sdk_kwargs:
        extra_body["reasoning"] = sdk_kwargs.pop("reasoning")
    if extra_body:
        sdk_kwargs["extra_body"] = extra_body
    return sdk_kwargs


def fill_openai_compatible(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str,
    base_url: str | None = None,
    max_workers: int = 8,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_create_kwargs: dict[str, Any] | None = None,
    max_attempts: int = 3,
    client_factory: Callable[..., Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FillResult:
    """Concurrently fetch and cache all cache-miss prompts.

    ``client_factory`` lets tests inject a fake client; by default a plain
    ``openai.OpenAI(api_key, base_url)`` is used (thread-safe across workers).
    """
    if client_factory is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = client_factory(api_key=api_key, base_url=base_url)

    # Partition into hits (already cached) and misses.
    misses: list[tuple[str, dict[str, Any], str]] = []
    hits = 0
    for p in prompts:
        kw = _create_kwargs(p, model, max_output_tokens, response_format, extra_create_kwargs)
        key = request_key(kw)
        if cache_lookup(cache_dir, key):
            hits += 1
        else:
            misses.append((p, kw, key))

    filled = errors = 0
    done = 0

    def response_text(resp: Any) -> str:
        if not getattr(resp, "choices", None):
            return ""
        return str(getattr(resp.choices[0].message, "content", None) or "").strip()

    def work(item: tuple[str, dict[str, Any], str]) -> bool:
        _p, kw, key = item
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.chat.completions.create(**_openai_sdk_kwargs(kw))
                if response_text(resp):
                    # n=1 -> exactly one choice; store the ChatCompletion as the cache entry,
                    # the same object the online OpenAI path caches.
                    cache_store(cache_dir, key, resp)
                    return True
                last_error = RuntimeError("provider returned an empty message")
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
                if errors <= 5:  # surface the first few; avoid log spam at scale
                    print(f"[fill] error: {exc}", flush=True)
            if progress:
                progress(done, len(misses))

    return FillResult(total=len(prompts), hits=hits, filled=filled, errors=errors)
