# Contributing to the Framework (T2.2.1)

This document is for contributors adding a new model adapter, a new prompt template, or a new config.

## The golden rule

**Do not modify the runner to add new models, new prompts, or new provider params.** The runner is the stable contract. Everything pluggable is plugged in elsewhere:

- New models -> add an adapter under `src/riddles_framework/models/`
- New prompts -> drop a YAML file in `src/riddles_framework/prompt_templates/`
- New provider params (top_p, frequency_penalty, etc.) → use `model_params:` in your run config YAML
- New docs or guides → add a Markdown file under `docs/`

If you want to edit `runner.py`, discuss it with the team lead first.

## Adding a new provider adapter

Three steps.

### 1. Write the adapter

Create `src/riddles_framework/models/<your_provider>_adapter.py`. Inherit from `riddles_framework.models.base.Model`. Implement `generate(...)` with this signature:

```python
def generate(
    self,
    prompt: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int = 256,
    seed: int | None = None,
    reasoning_mode: str | None = None,
    search_enabled: bool = False,
    response_format: str = "json",
    model_params: dict[str, Any] | None = None,
) -> GenerationResult: ...
```

Minimum responsibilities:

- Read `api_key_env_var` from `cfg` and look up the env var via `getenv_required`.
- Construct your provider's SDK client via the matching `make_*_client` in `cache_clients.py` so CacheSaver works transparently.
- **Only pass `temperature` to the provider when it is not `None`.** Reasoning models can degenerate at low temperature; relying on provider defaults is the recommended behavior.
- **Merge `model_params` LAST** into your request kwargs so user-supplied overrides win.
- Return a `GenerationResult` with `raw_response`, `status`, token counts if available, `latency_ms`, and useful `provider_details`.

Status assignment pattern (use this consistently across adapters):

```python
status = GenerationStatus.SUCCESS
if text is None or str(text).strip() == "":
    status = GenerationStatus.EMPTY
elif looks_like_refusal(text):
    status = GenerationStatus.REFUSAL
```

Errors should be caught inside `generate` and returned as `GenerationStatus.ERROR` with `error_message` populated, not raised. The runner depends on every adapter returning a `GenerationResult` rather than throwing.

### 2. Register it

Open `cli.py` and add a `(key, factory)` entry in `register_builtin_adapters`. Keys are `"{provider}:{access_pattern}"`:

```python
(("newprovider:direct_api", lambda spec, cfg: NewProviderAdapter(spec, cfg)),)
```

For working examples, see the adapters in `src/riddles_framework/models/`:
- `openai_adapter.py` — OpenAI direct API
- `anthropic_adapter.py` — Anthropic direct API, includes extended-thinking handling
- `gemini_adapter.py` — Google Gemini via `google-genai`
- `openrouter_adapter.py` — routes any OpenRouter-supported model through the OpenAI SDK; the most general-purpose pattern if your provider exposes an OpenAI-compatible endpoint
- `transformers_local_adapter.py` — local inference via `transformers.pipeline`

### 3. Add a YAML config

Drop a YAML under `configs/`. Use an existing config as the template. Be honest with `capabilities` — the runner's upfront capability check uses it to fail fast on configurations the model can't actually serve.

## Adding a new prompt template

Drop a YAML file into `src/riddles_framework/prompt_templates/`:

```yaml
template_id: my_template
template_version: v1
recommended_response_format: json # optional
instruction_text: |
  Your instruction text here.
```

Then in your run config:

```yaml
prompt_template_id: my_template
prompt_template_version: v1
```

No code change is required.

`recommended_response_format` is informational only — it does NOT override the run config's `response_format`. It exists so reviewers reading your template can see what format it was designed for. Document the linkage in your template's filename or in a brief comment.

## Adding a new provider parameter

Don't add a new `RunConfig` field. Use `model_params:` in the YAML:

```yaml
model_params:
  top_p: 0.95
  frequency_penalty: 0.1
  presence_penalty: 0.0
```

These flow through the adapter's `model_params` kwarg and are merged into the provider's request last. If a particular adapter needs special handling for a key (for example, Gemini's `config=` structure), put that branching inside the adapter, not in the runner.

## Capability semantics

The runner refuses to run if a config requests something the model doesn't support. So `capabilities` has to match reality:

- `supports_reasoning`: adapter handles `reasoning_mode` and the underlying model exposes a reasoning knob
- `supports_search`: adapter handles `search_enabled` and the underlying model can do web search
- `supports_structured_output`: adapter handles `response_format=json` / `json_schema`
- `supports_batch`: adapter has a real batch path (not just a for-loop)
- `supports_seed`: adapter passes `seed` through to the provider

If you're unsure, set it to `false`. False is the safe default; you get the upfront capability error and can change the value after you verify support.

## CacheSaver

If your provider has a CacheSaver-supported client, use the matching `make_*_client` helper. The pattern is already wired:

```python
self.client = make_openai_client(cfg.get("cache_enabled", False), **client_kwargs)
```

When `cache_enabled` is `false`, you get the raw SDK client. When `true`, you get the CacheSaver-wrapped equivalent. Either way the call surface (`.chat.completions.create(...)`, `.messages.create(...)`, etc.) is the same.

The same pattern applies to the other providers — `make_anthropic_client`, `make_gemini_client`, `make_openrouter_client`, and `make_transformers_client` are all in `cache_clients.py`. If you're adding a new provider that goes through OpenRouter or any OpenAI-compatible endpoint, `make_openrouter_client` already handles it; just pass the right `base_url`.

## Style notes

- Use `from __future__ import annotations` at the top of every file.
- Keep adapters small. If a file grows past ~150 lines, the work probably belongs somewhere else.
- Don't add provider-specific logic to `runner.py`, `prompts.py`, `schemas.py`, or `writers.py`. Those are shared. Adapters are not.

## Testing your adapter

The framework ships with a dummy adapter (`models/dummy.py`). Use the dummy config as your scaffold and confirm the end-to-end loop runs cleanly first. Then swap in your adapter via a new YAML.

Don't add a network-dependent test to `tests/test_framework.py`. Keep the existing suite hermetic.

## When in doubt

Ask questions in `#multicultural-riddles-benchmark`
