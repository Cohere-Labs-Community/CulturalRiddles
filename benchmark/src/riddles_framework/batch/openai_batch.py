"""OpenAI Batch API backend (CacheSaver-compatible).

Fills CacheSaver's on-disk cache via OpenAI's offline Batch API (~50% cheaper,
high throughput, async). Flow:

  build JSONL of /v1/chat/completions requests for cache misses
    -> files.create(purpose="batch") -> batches.create
    -> poll batches.retrieve until completed
    -> download output, parse each line into a ChatCompletion
    -> cache_store(key, completion)

The per-request ``custom_id`` IS the CacheSaver key, so results map straight
back. The cached entry is an ``openai`` ``ChatCompletion`` — identical to what
the online OpenAI path stores (verified) — so a later ``cache_enabled`` run
reads every prompt as a hit.

Steps that need the live API (submit / poll / download) are separated from the
pure functions (request build, result write-back) so the latter are unit-tested
without a real batch job.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .cache_compat import cache_lookup, cache_store, openai_request_key


@dataclass
class BatchPlan:
    requests: list[dict[str, Any]]  # JSONL lines for the misses
    hit_count: int


def build_batch_requests(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_body: dict[str, Any] | None = None,
) -> BatchPlan:
    """Build one Batch line per cache-miss prompt; skip prompts already cached.

    ``body`` mirrors the OpenAI adapter's ``create`` kwargs so the key (and thus
    the eventual cache hit) matches. ``custom_id`` is the CacheSaver key.
    """
    requests: list[dict[str, Any]] = []
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
        requests.append(
            {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
    return BatchPlan(requests=requests, hit_count=hits)


def writeback_output(cache_dir: str, output_jsonl: str) -> tuple[int, int]:
    """Parse Batch output JSONL into ChatCompletions and store them. -> (ok, err)."""
    from openai.types.chat import ChatCompletion

    ok = err = 0
    for line in output_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = rec.get("custom_id")
        try:
            body = rec["response"]["body"]
            completion = ChatCompletion.model_validate(body)
            cache_store(cache_dir, key, completion)
            ok += 1
        except Exception:
            err += 1
    return ok, err


def run_openai_batch(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str,
    base_url: str | None = None,
    max_output_tokens: int = 4096,
    response_format: str = "free_form",
    extra_body: dict[str, Any] | None = None,
    completion_window: str = "24h",
    poll_seconds: int = 30,
    max_wait_seconds: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """End-to-end: submit a Batch job for misses, poll, write results to cache."""
    from openai import OpenAI

    plan = build_batch_requests(
        prompts=prompts,
        model=model,
        cache_dir=cache_dir,
        max_output_tokens=max_output_tokens,
        response_format=response_format,
        extra_body=extra_body,
    )
    log(f"[openai-batch] {plan.hit_count} already cached, {len(plan.requests)} to submit")
    if not plan.requests:
        return {"submitted": 0, "cached": plan.hit_count, "written": 0, "errors": 0}

    client = OpenAI(api_key=api_key, base_url=base_url)
    buf = io.BytesIO("\n".join(json.dumps(r) for r in plan.requests).encode("utf-8"))
    buf.name = "batch_input.jsonl"
    in_file = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    log(f"[openai-batch] submitted batch {batch.id}; polling every {poll_seconds}s")

    waited = 0
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.batches.retrieve(batch.id)
        log(
            f"[openai-batch] {batch.id} status={batch.status} "
            f"({getattr(batch.request_counts, 'completed', '?')}/"
            f"{getattr(batch.request_counts, 'total', '?')})"
        )
        if max_wait_seconds is not None and waited >= max_wait_seconds:
            return {"batch_id": batch.id, "status": batch.status, "timed_out": True}

    if batch.status != "completed":
        return {"batch_id": batch.id, "status": batch.status, "written": 0}

    output = client.files.content(batch.output_file_id).text
    ok, err = writeback_output(cache_dir, output)
    log(f"[openai-batch] wrote {ok} to cache ({err} errors)")
    return {
        "batch_id": batch.id,
        "status": "completed",
        "cached": plan.hit_count,
        "written": ok,
        "errors": err,
    }
