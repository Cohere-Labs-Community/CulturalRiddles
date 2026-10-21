"""CacheSaver-compatible direct access to the on-disk cache.

CacheSaver's ``SyncWrapper`` runs an asyncio pipeline (per-key locks, an online
batcher) and calls ``asyncio.run`` per request — it is serial and NOT safe to
call from multiple threads. To fill the cache concurrently (thread pool) or via
a provider Batch API, we instead read/write the underlying ``diskcache.Cache``
ourselves, reproducing CacheSaver's exact key and entry format so a normal
``cache_enabled`` run reads our entries as plain cache hits.

Validated against a real run's cache: ``request_key`` reproduces the key
CacheSaver stored, and entries are the provider response objects the online
``response_adapter`` reads (an ``openai`` ``ChatCompletion`` for OpenAI-compatible
providers).

Key facts that make this safe and stable:
- CacheSaver's ``hash(request)`` excludes ``n``, ``request_id`` and ``namespace``
  (see ``cachesaver/typedefs.py``), so a batch request (n>1) and an online
  request (n=1) for the same prompt collide on the same key.
- A cache value is a ``list`` of response entries; the online reader takes the
  next ``n`` unused entries. For a fresh miss we store exactly one entry, which
  the online run (n=1, used=0) returns directly.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

# Import lazily-safe: cachesaver is only required when caching is in play.
from cachesaver.typedefs import Request
from cachesaver.typedefs import hash as _cs_hash
from diskcache import Cache


def openai_request_key(create_kwargs: dict[str, Any]) -> str:
    """CacheSaver key for providers using the DEFAULT hash.

    Covers every OpenAI-compatible provider (OpenAI, Cohere, OpenRouter,
    Together) and Anthropic — they all use ``cachesaver.typedefs.hash``.
    ``create_kwargs`` must match exactly what the adapter passes to its SDK
    ``create`` call (``model`` / ``messages`` / ``max_tokens`` /
    ``max_completion_tokens`` / ``response_format`` / ...). ``n`` /
    ``request_id`` / ``namespace`` don't affect the key.
    """
    req = Request(args=(), kwargs=dict(create_kwargs), n=1, request_id="x", namespace=None)
    return _cs_hash(req)


# Back-compat alias.
request_key = openai_request_key


def gemini_request_key(call_kwargs: dict[str, Any]) -> str:
    """CacheSaver key for Gemini, which uses a CUSTOM hash.

    Mirrors ``cachesaver.models.gemini.GeminiModel.hash``: hash over the
    request minus ``n`` / ``request_id`` / ``namespace``, with
    ``config.candidate_count`` nulled when a ``config`` is present.
    ``call_kwargs`` must match the Gemini adapter's
    ``models.generate_content(**kwargs)`` (``model`` / ``contents`` / [``config``]).
    """
    from deepdiff import DeepHash

    req = Request(args=(), kwargs=dict(call_kwargs), n=1, request_id="x", namespace=None)
    params = asdict(req)
    del params["request_id"]
    del params["namespace"]
    del params["n"]
    if "config" in params["kwargs"]:
        try:
            params["kwargs"]["config"].candidate_count = None
        except Exception:
            pass
    return DeepHash(params)[params]


def cache_lookup(cache_dir: str, key: str) -> list[Any] | None:
    """Return the cached entry list for ``key``, or ``None`` if absent/empty."""
    with Cache(cache_dir) as c:
        entries = c.get(key, None)
    return entries or None


def cache_store(cache_dir: str, key: str, entry: Any) -> None:
    """Store a single response ``entry`` under ``key`` for a fresh miss.

    No-op if the key already has entries, so this never clobbers results an
    online run produced. ``entry`` must be the same provider response object the
    online path caches (e.g. an ``openai`` ``ChatCompletion`` with one choice).
    """
    with Cache(cache_dir) as c:
        if c.get(key, None):
            return
        c.set(key, [entry])
