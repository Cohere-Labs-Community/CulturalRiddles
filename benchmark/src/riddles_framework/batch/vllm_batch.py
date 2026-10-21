"""vLLM Concurrent Async Batch backend (CacheSaver-compatible).

Sends concurrent HTTP requests to standard /v1/chat/completions.
vLLM automatically handles continuous batching on the server side.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .cache_compat import cache_lookup, cache_store, openai_request_key


@dataclass
class VLLMBatchPlan:
    items: list[tuple[str, str]]  # list of (cache_key, prompt_string)
    hit_count: int


def build_vllm_batch_requests(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_body: dict[str, Any] | None = None,
) -> VLLMBatchPlan:
    items: list[tuple[str, str]] = []
    hits = 0
    seen: set[str] = set()

    for p in prompts:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "max_completion_tokens": max_output_tokens,
        }
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        if extra_body:
            body.update(extra_body)

        key = openai_request_key(body)
        if key in seen:
            continue
        seen.add(key)

        if cache_lookup(cache_dir, key):
            hits += 1
            continue

        items.append((key, p))

    return VLLMBatchPlan(items=items, hit_count=hits)


async def _process_single_prompt(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    headers: dict[str, str],
    key: str,
    prompt: str,
    model: str,
    max_output_tokens: int,
    response_format: str,
    extra_body: dict[str, Any] | None,
    cache_dir: str,
) -> bool:
    from openai.types.chat import ChatCompletion

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_output_tokens,
    }
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    if extra_body:
        payload.update(extra_body)

    async with semaphore:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            completion = ChatCompletion.model_validate(data)
            cache_store(cache_dir, key, completion)
            return True
        except Exception:
            return False


def run_vllm_batch(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str = "EMPTY",
    base_url: str | None = None,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_body: dict[str, Any] | None = None,
    max_concurrency: int = 64,  # Adjust higher or lower depending on GPU memory
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Sends concurrent requests to /v1/chat/completions."""
    base = base_url or "http://localhost:8000/v1"
    endpoint = f"{base.rstrip('/')}/chat/completions"

    plan = build_vllm_batch_requests(
        prompts=prompts,
        model=model,
        cache_dir=cache_dir,
        max_output_tokens=max_output_tokens,
        response_format=response_format,
        extra_body=extra_body,
    )
    log(f"[vllm-batch] {plan.hit_count} already cached, {len(plan.items)} to request online")

    if not plan.items:
        return {"submitted": 0, "cached": plan.hit_count, "written": 0, "errors": 0}

    headers = {"Authorization": f"Bearer {api_key}"}

    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrency)
        async with httpx.AsyncClient(timeout=None) as client:
            tasks = [
                _process_single_prompt(
                    client,
                    semaphore,
                    endpoint,
                    headers,
                    key,
                    p,
                    model,
                    max_output_tokens,
                    response_format,
                    extra_body,
                    cache_dir,
                )
                for key, p in plan.items
            ]
            return await asyncio.gather(*tasks)

    start_time = time.time()
    results = asyncio.run(_run_all())
    latency = int(time.time() - start_time)

    ok = sum(1 for r in results if r)
    err = sum(1 for r in results if not r)

    log(f"[vllm-batch] Batch completed in {latency}s. Wrote {ok} to cache ({err} errors)")

    return {"status": "completed", "cached": plan.hit_count, "written": ok, "errors": err}
