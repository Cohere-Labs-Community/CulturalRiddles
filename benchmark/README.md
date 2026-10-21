# Benchmark

The `benchmark/` directory contains the model runner, provider configurations, and implementation guides. The paper's model configurations were run with this package; their raw responses are in `data/canonical/model_responses.parquet`.

Run commands from the repository root. Install the root project with `uv sync`. For local Transformers adapters, install the optional dependencies with `uv sync --extra local`; loading a model might require a model download.

## Commands and source map

You do not need to navigate into `src/` or use internal Python module paths. Install once with `uv sync --frozen`, then run either command from the repository root:

```bash
uv run --frozen riddles-framework --help
uv run --frozen riddles-judge --help
```

| Task | Public entrypoint | Implementation |
| --- | --- | --- |
| Generate model answers | `riddles-framework` | [Model runner](src/riddles_framework/runner.py) |
| Evaluate answers with a judge | `riddles-judge` | [Judge runner](src/riddles_framework/judging/llm_judge_run.py) |
| Inspect the judge rubric and structured verdict | No inference needed | [Judge prompt and schema](src/riddles_framework/judging/llm_as_a_judge.py) |
| Reproduce recorded judge-study results | `python -m analysis.run_all --stage judges` | [Offline analysis guide](../analysis/llm_judge/README.md) |

## Credentials and provider routing

Offline reproduction, submission checks, and the dummy adapter need no credentials. For paid calls, copy the root template without overwriting an existing `.env`:

```bash
cp -n .env.template .env
```

Fill only the keys needed for your run. `.env` is ignored by Git and excluded from release archives; `.env.template` contains blank values and is safe to distribute. Never put credentials in YAML configurations, tracked files, or run notes.

Judges call provider APIs directly through the LiteLLM Python client, not OpenRouter or a LiteLLM proxy:

| Judge | Provider request | Credential |
| --- | --- | --- |
| `gemini-3.5-flash` | Google's Gemini API via `gemini/gemini-3.5-flash` | `GEMINI_API_KEY` |
| `deepseek-v4.1-flash` | `https://api.deepseek.com/beta` via `deepseek/deepseek-flash` | `DEEPSEEK_API_KEY` |

The judge runner loads the repository-root `.env` automatically without overriding existing shell variables. DeepSeek uses the beta endpoint's strict verdict tool; Gemini uses native structured output. The request model, provider, and settings are recorded in the run manifest. Historical aliases might resolve differently over time.

Solver routing is independent: the YAML's `model_spec.provider`, `access_pattern`, `base_url`, and `api_key_env_var` select its adapter, endpoint, and credential. Some released solver configurations use OpenRouter and require `OPENROUTER_API_KEY`; that key doesn't authenticate either judge. To load `.env` for a solver run, use the following command:

```bash
uv run --frozen --env-file .env riddles-framework \
  --config benchmark/configs/benchmark_v0/CONFIG.yaml
```

Replace `CONFIG.yaml` with the selected configuration file. Solver commands do not load `.env` themselves; omit `--env-file` if you already export the required key in your shell.

## Run an offline smoke test

The dummy adapter makes no network requests and doesn't download a model. Run five canonical riddles:

```bash
uv run riddles-framework --config benchmark/configs/dummy.yaml --limit 5
```

The command prints a run ID and writes results under `data/runs/RUN_ID/`. The run contains `results.jsonl` (one result per riddle), `run_config.json`, `run_metadata.json`, and `manifest.json`. To run a real model, copy a configuration from `benchmark/configs/benchmark_v0/`, set `model_spec.provider_model_name`, export the key named by `api_key_env_var`, and run the CLI with that configuration. Provider calls might incur charges.

## Score a new run with released evidence

Scoring applies explicit answer rules first, then the valid first human annotation for a matching `(riddle_id, normalized answer)` from `data/canonical/human_eval/annotations.parquet`, then recorded zero-shot judge verdicts from canonical setting C when both judges agree. Unmatched or unresolved answers remain unscored; they aren't treated as incorrect. The scorer writes only to `data/derived/` or `data/runs/`, never to canonical evidence or frozen benchmark scores.

After a run, replace `RUN_ID` with its directory name and `MODEL_KEY` with a stable model key for analysis. Use an existing key from `data/canonical/model_mapping.csv` only when the new run represents that recorded model. Save scores in the run directory:

```bash
uv run python -m analysis.scoring \
  --responses data/runs/RUN_ID/results.jsonl \
  --model MODEL_KEY \
  --out data/runs/RUN_ID/scored.parquet
```

This also writes `data/runs/RUN_ID/scored_unresolved.parquet` for answers that need further adjudication. The model key is a label for analysis; it does not change the raw run or imply that it matches a historical model. The scoring command reads the frozen canonical human annotations and judge verdicts as evidence. A new workbook is not consumed by this command: validate submitted workbooks with the [human evaluation guide](../human_eval/README.md), and use the repository's reviewed evidence-release workflow before treating new annotations as canonical.

To prepare remaining judge work without making provider calls, inspect the unresolved export with the following command:

```bash
uv run riddles-judge \
  data/runs/RUN_ID/scored_unresolved.parquet \
  --model gemini-3.5-flash \
  --out data/runs/llm_judge/RUN_ID-preview.jsonl --dry-run
```

A dry run creates a frozen `.inputs.jsonl` request snapshot and `.manifest.json` manifest; it doesn't call a judge. It enriches unresolved rows with riddle context and the public human-review rubric. If the unresolved export is empty, no judge work is needed.

## Complete a separate paid judge evaluation

The following commands make paid provider calls. Do not run them for an offline smoke test or paper reproduction. Set `GEMINI_API_KEY` and `DEEPSEEK_API_KEY` in the root `.env` or your shell as described in [credential setup](#credentials-and-provider-routing).

Use an unused output filename for each judge, distinct from the dry-run preview:

```bash
uv run riddles-judge \
  data/runs/RUN_ID/scored_unresolved.parquet --model gemini-3.5-flash \
  --out data/runs/llm_judge/RUN_ID-gemini.jsonl

uv run riddles-judge \
  data/runs/RUN_ID/scored_unresolved.parquet --model deepseek-v4.1-flash \
  --out data/runs/llm_judge/RUN_ID-deepseek.jsonl
```

The runner accepts output paths only beneath `data/runs/`. It freezes inputs and request configuration before inference, validates verdicts against the rubric, and keeps technical failures separate from answer scores. To resume an interrupted paid run, repeat its exact command with `--resume`; changes to frozen inputs or request configuration invalidate resume compatibility.

After both paid runs finish, apply their validated final verdicts to the original responses:

```bash
uv run python -m analysis.scoring \
  --responses data/runs/RUN_ID/results.jsonl --model MODEL_KEY \
  --verdicts data/runs/llm_judge/RUN_ID-gemini.jsonl \
             data/runs/llm_judge/RUN_ID-deepseek.jsonl \
  --out data/runs/RUN_ID/scored_with_judges.parquet
```

Use the same `MODEL_KEY` as in the first scoring command. Two agreeing zero-shot scores complete an otherwise unresolved answer. Missing, failed, or disagreeing verdicts remain unscored in `scored_with_judges_unresolved.parquet` for further review, not zero-scored. This separate evaluation does not replace the paper's recorded verdicts. For analysis of those records, see the [offline judge guide](../analysis/llm_judge/README.md).

The frozen benchmark scores remain in `data/canonical/benchmark_results.parquet`. Model runs and new-run scores don't modify them. For the original paper results, see the [model audit](docs/model_audit.md).

## Additional guides

For more information, see the following guides:

- [Getting started](docs/getting-started.md): provider configurations and output details.
- [Configuration reference](configs/benchmark_v0/README.md): configuration fields.
- [Batch guide](src/riddles_framework/batch/README.md): provider concurrency and Batch APIs.
- [Code reference](docs/code-reference.md): runner behavior and artifacts.
- [Contributing](CONTRIBUTING.md): provider adapters and prompt templates.
