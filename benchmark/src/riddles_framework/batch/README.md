# Getting results faster and cheaper

The runner sends requests **one at a time**. This package adds faster
paths. All of them share one CacheSaver-compatible cache layer, so results are
interchangeable with a normal `cache_enabled` run.

## Pick a path

| Goal | Providers | How |
| --- | --- | --- |
| **Concurrency inside a normal run** | Cohere, vLLM, and OpenRouter (OpenAI-compatible) | Set `batch_mode: true` and `batch_size: N` in the config. |
| **Pre-fill the cache concurrently**, then run | Cohere, vLLM, OpenRouter, Together | `python -m riddles_framework.batch.fill --workers N` |
| **Offline Batch API (~50% cheaper)** | OpenAI, Anthropic, Gemini | `python -m riddles_framework.batch.fill --backend batch` |

Concurrency gives a speedup of up to N times, bounded by the provider's rate limit. The
Batch APIs are offline (minutes to hours) but cheaper and high-throughput. Cohere, vLLM,
and OpenRouter have **no chat Batch API**, so concurrency is their fast path.

## How it works

CacheSaver's `SyncWrapper` runs an asyncio pipeline (per-key locks, an online
batcher) through `asyncio.run` per call. It is serial and **not thread-safe**
(verified: concurrent threads deadlock). So everything here **bypasses the
wrapper** and reads and writes the underlying `diskcache` directly through
`cache_compat.py`, reproducing CacheSaver's exact key and per-provider entry
format. Verified against real run caches: the reconstructed key matches, and a
live CacheSaver client reads these entries as plain hits.

This design has the following consequences:

- A fast run populates the cache. A later normal run reads every prompt as a hit
  (no re-billing), so the two are fully interchangeable.
- Only cache **misses** are fetched. Changing the prompt, model, or params changes the
  key, so a different config re-fills.

---

## Path 1 — `batch_mode` (concurrency inside the run)

For OpenAI-compatible providers (Cohere, OpenRouter), the adapter's
`generate_batch` runs `batch_size` requests concurrently through a plain
thread-safe client, reading and writing the cache directly. Configure it as follows:

```yaml
batch_mode: true
batch_size: 12          # concurrency
model_spec:
  capabilities:
    supports_batch: true   # required, else the runner refuses batch_mode
```

Then run the CLI normally, with no extra step:

```bash
uv run python -m riddles_framework.cli --config benchmark/configs/benchmark_v0/cohere_tiny_aya.yaml
```

This path is implemented in `models/concurrent_batch.py` and the `generate_batch` overrides in
`cohere_adapter.py` and `openrouter_adapter.py`.

## Path 2 — `batch.fill` (pre-fill, then run)

This path fills the cache out-of-band, and then a normal run reads hits. It has two backends, auto-
selected by provider (override with `--backend`). Run the following commands:

```bash
uv run python -m riddles_framework.batch.fill \
  --config benchmark/configs/benchmark_v0/cohere_tiny_aya.yaml --workers 12

# OpenAI -> provider Batch API
uv run python -m riddles_framework.batch.fill \
  --config benchmark/configs/benchmark_v0/gpt55_none.yaml --backend batch

# Count misses without making requests
uv run python -m riddles_framework.batch.fill \
  --config benchmark/configs/benchmark_v0/gpt55_none.yaml --dry-run

# Run the configuration from the repository root
uv run riddles-framework \
  --config benchmark/configs/benchmark_v0/gpt55_none.yaml
```

The flags are `--backend auto|threadpool|batch`, `--workers N` (threadpool concurrency),
`--poll N` (batch poll interval, in seconds), and `--dry-run`.

---

## Provider support

| Provider | Concurrency (`batch_mode` / `--workers`) | Batch API (`--backend batch`) |
| --- | --- | --- |
| OpenAI | Use the Batch API | Yes |
| Anthropic | Use the Batch API | Yes |
| Gemini | Use the Batch API | Yes |
| Cohere | Yes | No chat Batch API |
| OpenRouter | Yes | No |
| Together | Yes, with `--workers` | No |

## Cache key and entry format

| Provider | Cache key | Cache entry |
| --- | --- | --- |
| OpenAI / Cohere / OpenRouter / Together | default hash | `openai…ChatCompletion` |
| Anthropic | default hash | `anthropic…Message` |
| Gemini | custom hash (≡ default when no `config`) | `google…GenerateContentResponse` |

## Files

| File | Role |
| --- | --- |
| `cache_compat.py` | Shared diskcache read/write and per-provider keys. |
| `threadpool.py` | Concurrent pre-fill for OpenAI-compatible providers. |
| `openai_batch.py` / `anthropic_batch.py` / `gemini_batch.py` | Provider Batch APIs. |
| `fill.py` | CLI driver (`python -m riddles_framework.batch.fill`). |
| `../models/concurrent_batch.py` | In-run `batch_mode` concurrency (used by the adapters). |
