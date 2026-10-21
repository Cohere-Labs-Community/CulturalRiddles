"""Anthropic Message Batches backend (CacheSaver-compatible).

Uses Anthropic's Message Batches API to fill cache misses. The cached entry is
an ``anthropic`` ``Message`` (what the online Anthropic path stores), keyed by
the default CacheSaver hash over the ``messages.create`` kwargs.

The Anthropic adapter sends ``{model, max_tokens, messages, [thinking,
temperature]}``; this backend replicates the non-reasoning shape
``{model, max_tokens, messages}``. If you batch a reasoning (thinking) run, pass
the same extra params via ``extra_params`` so the key matches.

Submit/poll/results need the live API; request-build and write-back are pure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .cache_compat import cache_lookup, cache_store, openai_request_key


@dataclass
class BatchPlan:
    requests: list[dict[str, Any]]
    hit_count: int


def build_batch_requests(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    max_output_tokens: int = 4096,
    extra_params: dict[str, Any] | None = None,
) -> BatchPlan:
    requests: list[dict[str, Any]] = []
    hits = 0
    seen: set[str] = set()
    for p in prompts:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": p}],
        }
        if extra_params:
            params.update(extra_params)
        key = openai_request_key(params)  # Anthropic uses the default hash
        if key in seen:
            continue
        seen.add(key)
        if cache_lookup(cache_dir, key):
            hits += 1
            continue
        requests.append({"custom_id": key, "params": params})
    return BatchPlan(requests=requests, hit_count=hits)


def writeback_results(cache_dir: str, results) -> tuple[int, int]:
    """Store succeeded ``Message`` results. ``results`` iterates entries with
    ``.custom_id`` and ``.result`` (``.type`` / ``.message``)."""
    ok = err = 0
    for r in results:
        try:
            if r.result.type == "succeeded":
                cache_store(cache_dir, r.custom_id, r.result.message)
                ok += 1
            else:
                err += 1
        except Exception:
            err += 1
    return ok, err


def run_anthropic_batch(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str,
    max_output_tokens: int = 4096,
    extra_params: dict[str, Any] | None = None,
    poll_seconds: int = 30,
    max_wait_seconds: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from anthropic import Anthropic

    plan = build_batch_requests(
        prompts=prompts,
        model=model,
        cache_dir=cache_dir,
        max_output_tokens=max_output_tokens,
        extra_params=extra_params,
    )
    log(f"[anthropic-batch] {plan.hit_count} cached, {len(plan.requests)} to submit")
    if not plan.requests:
        return {"submitted": 0, "cached": plan.hit_count, "written": 0}

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=plan.requests)
    log(f"[anthropic-batch] submitted {batch.id}; polling every {poll_seconds}s")

    waited = 0
    while batch.processing_status != "ended":
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.messages.batches.retrieve(batch.id)
        log(f"[anthropic-batch] {batch.id} status={batch.processing_status}")
        if max_wait_seconds is not None and waited >= max_wait_seconds:
            return {"batch_id": batch.id, "status": batch.processing_status, "timed_out": True}

    ok, err = writeback_results(cache_dir, client.messages.batches.results(batch.id))
    log(f"[anthropic-batch] wrote {ok} to cache ({err} errors)")
    return {
        "batch_id": batch.id,
        "status": "ended",
        "cached": plan.hit_count,
        "written": ok,
        "errors": err,
    }
