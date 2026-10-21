"""Gemini Batch backend (CacheSaver-compatible).

The cache entry is a google ``GenerateContentResponse`` keyed by
``gemini_request_key({"model": ..., "contents": prompt})`` (matches the online
Gemini path, confirmed against a real run's cache).

Validated live end-to-end against **google-genai 2.8.0** (inline ``src``
requests, ``JOB_STATE_*`` polling, ``dest.inlined_responses``). That SDK's batch
surface changes between versions — if you upgrade ``google-genai``, re-check
``batches.create`` and the ``dest.inlined_responses`` unwrapping here.

Responses are matched back to requests by a per-request ``cache_key`` carried in
request metadata — NOT by submission order (Gemini does not guarantee inline
response ordering). If a backend doesn't echo inline metadata, ``run_gemini_batch``
fails loudly rather than mis-store; switch to file-based batch (a ``key`` per
JSONL line) there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .cache_compat import cache_lookup, cache_store, gemini_request_key

_KEY = "cache_key"  # request metadata field used to correlate responses


@dataclass
class BatchPlan:
    items: list[tuple[str, str]]  # (cache_key, prompt) per miss
    hit_count: int


def build_batch_plan(*, prompts: list[str], model: str, cache_dir: str) -> BatchPlan:
    items: list[tuple[str, str]] = []
    hits = 0
    seen: set[str] = set()
    for p in prompts:
        key = gemini_request_key({"model": model, "contents": p})
        if key in seen:
            continue
        seen.add(key)
        if cache_lookup(cache_dir, key):
            hits += 1
            continue
        items.append((key, p))
    return BatchPlan(items=items, hit_count=hits)


def _response_key(inlined) -> str | None:
    """Pull the request's ``cache_key`` back out of an InlinedResponse's metadata."""
    md = getattr(inlined, "metadata", None)
    if md is None:
        return None
    if isinstance(md, dict):
        return md.get(_KEY)
    try:
        return dict(md).get(_KEY)
    except Exception:
        return getattr(md, _KEY, None)


def writeback_responses(cache_dir: str, plan: BatchPlan, inlined_responses) -> dict[str, int]:
    """Store each response under the key carried in ITS OWN metadata.

    Gemini does NOT guarantee inline responses come back in submission order, so
    we match by the per-request ``cache_key`` metadata — never by position. A
    response whose metadata doesn't identify a known request is left unmatched
    (counted, never guessed), so results can't land under the wrong key.
    """
    expected = {k for k, _ in plan.items}
    ok = err = unmatched = 0
    for ir in inlined_responses:
        key = _response_key(ir)
        if key not in expected:
            unmatched += 1
            continue
        if getattr(ir, "error", None) is not None or getattr(ir, "response", None) is None:
            err += 1
            continue
        try:
            cache_store(cache_dir, key, ir.response)
            ok += 1
        except Exception:
            err += 1
    return {"written": ok, "errors": err, "unmatched": unmatched}


def run_gemini_batch(
    *,
    prompts: list[str],
    model: str,
    cache_dir: str,
    api_key: str,
    poll_seconds: int = 30,
    max_wait_seconds: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from google import genai

    plan = build_batch_plan(prompts=prompts, model=model, cache_dir=cache_dir)
    log(f"[gemini-batch] {plan.hit_count} cached, {len(plan.items)} to submit")
    if not plan.items:
        return {"submitted": 0, "cached": plan.hit_count, "written": 0}

    client = genai.Client(api_key=api_key)
    # Inlined requests carry the cache_key in metadata so responses (which are
    # NOT guaranteed to be in submission order) can be matched back, not zipped.
    src = [{"model": model, "contents": p, "metadata": {_KEY: k}} for k, p in plan.items]
    job = client.batches.create(model=model, src=src)
    log(f"[gemini-batch] submitted {job.name}; polling every {poll_seconds}s")

    waited = 0
    while not str(getattr(job, "state", "")).endswith(
        ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")
    ):
        time.sleep(poll_seconds)
        waited += poll_seconds
        job = client.batches.get(name=job.name)
        log(f"[gemini-batch] {job.name} state={getattr(job, 'state', '?')}")
        if max_wait_seconds is not None and waited >= max_wait_seconds:
            return {"job": job.name, "state": str(getattr(job, "state", "?")), "timed_out": True}

    if not str(getattr(job, "state", "")).endswith("SUCCEEDED"):
        return {"job": job.name, "state": str(getattr(job, "state", "?")), "written": 0}

    stats = writeback_responses(cache_dir, plan, job.dest.inlined_responses)
    log(
        f"[gemini-batch] wrote {stats['written']} to cache "
        f"({stats['errors']} errors, {stats['unmatched']} unmatched)"
    )
    # Safety: if nothing matched, the response metadata didn't round-trip — fail
    # loudly rather than risk mis-storing. Switch to file-based batch (a `key` per
    # JSONL line) if your backend doesn't echo inline metadata.
    if stats["written"] == 0 and stats["unmatched"] > 0:
        raise RuntimeError(
            "Gemini batch: no response carried matching request metadata "
            f"({stats['unmatched']} unmatched). Inline metadata correlation is "
            "unavailable on this backend/SDK — use file-based batch matching."
        )
    return {"job": job.name, "state": "SUCCEEDED", "cached": plan.hit_count, **stats}
