# Run configurations

These YAML files are provider examples for the canonical riddle corpus. They
are not a record of the historical request IDs; see
`data/canonical/model_runs.csv` for saved run provenance and
`data/canonical/model_mapping.csv` for canonical model names.

Run commands from the repository root. Provider configurations might incur
charges. For the offline dummy command, see the
[benchmark guide](../../README.md).

## Dataset and output paths

The configurations use `data/canonical/riddles.jsonl` and label the dataset
`canonical_v1`. Configuration paths are resolved from the repository root.
The CLI writes to `data/runs` by default; use `--output-root` to choose another
location.

## Provider settings

Set `model_spec.provider_model_name` to the exact model ID that your provider
account can call. Export the API key named by `api_key_env_var`. The provider
model ID is distinct from the canonical model key and display name in
`model_mapping.csv`.

The CLI rejects an unregistered provider/access-pattern pair before creating a
model or run directory. Historical entries in `model_runs.csv` are provenance,
not executable configurations. The historical `deepseek`
provider adapter isn't shipped, and no DeepSeek V4 Flash direct-provider example
is included. The bundled DeepSeek V4 Pro examples use the supported OpenRouter
adapter instead. To use `qwen3.5_vllm.yaml`, first start an OpenAI-compatible
server that serves its named model at the configured `base_url`.

`reasoning_mode` and `search_enabled` describe runner settings. They do not
prove that a provider received those settings. For example, the OpenRouter
adapter does not serialize `reasoning_mode`; only fields in `model_params`
reach its request. Read the selected adapter's serialization before describing
effort or search in an analysis.

The following table describes the configuration fields:

| Field | Meaning |
| --- | --- |
| `dataset_path` | JSON Lines dataset path, relative to the repository root. |
| `dataset_version` | Dataset label recorded in run metadata. |
| `prompt_template_id` / `prompt_template_version` | Prompt template identifier. The bundled template is `riddle_master` version `v1`. |
| `prompt_language` | Language of the instructions. |
| `riddle_language_presented` | `source` or `english`. |
| `is_hint_provided` | Whether the prompt includes a hint. |
| `reasoning_mode` | Adapter input for a reasoning setting. |
| `search_enabled` | Adapter input for search. |
| `model_params` | Additional provider-specific request parameters. |
| `batch_mode` / `batch_size` | Whether to run concurrent requests and the concurrency limit. |
| `api_key_env_var` | Name of the environment variable holding the provider key. |
| `base_url` | Optional provider-compatible API endpoint. |
