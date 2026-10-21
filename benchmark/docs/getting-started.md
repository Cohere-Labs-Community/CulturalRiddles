# Get started

Run commands from the repository root. Install the project with `uv sync`.

## Run an offline smoke test

The dummy adapter makes no network requests and doesn't download a model. Run it on five rows from the canonical corpus:

```bash
uv run riddles-framework --config benchmark/configs/dummy.yaml --limit 5
```

The run writes `results.jsonl`, its resolved configuration and metadata, and a manifest under `data/runs/RUN_ID/` by default. To score it against released human labels and judge verdicts, follow the [benchmark scoring workflow](../README.md#score-a-new-run-with-released-evidence).

## Run a provider configuration

1. Copy a configuration from `benchmark/configs/benchmark_v0/`.
2. Set `model_spec.provider_model_name` to a model ID that your provider account can call.
3. Export the API key named by `api_key_env_var` in the configuration.
4. Run the configuration from the repository root:

   ```bash
   uv run riddles-framework \
     --config benchmark/configs/benchmark_v0/CONFIG.yaml
   ```

Replace `CONFIG` with the configuration's file name. Provider runs can incur charges.

## Configure a run

The bundled configurations load `data/canonical/riddles.jsonl`. They use the `riddle_master` prompt template, English instructions, the riddle's source language, and no hint. Set `reasoning_mode` and `search_enabled` only when the selected adapter supports the requested option. Some adapters ignore `reasoning_mode`; inspect that adapter and `model_params` to determine the request that it sends.

Use `--limit N` to restrict a run to the first `N` dataset rows. Use `--output-root PATH` to override the default `data/runs` directory. Paths in run configurations are resolved from the repository root.

For configuration fields, see the [configuration reference](../configs/benchmark_v0/README.md). For output artifacts and runner behavior, see the [code reference](code-reference.md).
