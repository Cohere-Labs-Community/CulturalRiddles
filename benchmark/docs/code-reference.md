# Benchmark code reference

Run commands from the repository root. The root project installs the
`riddles-framework` command and loads its package from `benchmark/src/riddles_framework/`.

## Command-line interface

```text
riddles-framework --config CONFIG [--output-root PATH] [--limit N]
                  [--run-batch BATCH ...]
```

- `--config` is required. Configuration paths are relative to the repository
  root.
- `--output-root` defaults to `data/runs`.
- `--limit` limits the run to the first `N` rows after dataset filters. It must
  be a positive integer.
- `--run-batch` selects the named dataset batches. The canonical corpus doesn't
  include batch tags, so don't pass this option with the canonical corpus.

For an offline smoke test on five canonical riddles, run:

```bash
uv run riddles-framework --config benchmark/configs/dummy.yaml --limit 5
```

## Dataset contract

`dataset_path` points to JSON Lines data. Each row must include `riddle_id`,
`language`, `community`, `riddle_original`, and `reference_answer_original`.
The loader also reads `topic`, `is_wildcard`, English translations, hints,
notes, and optional batch tags. Unknown JSON fields are ignored.

The default configurations use `data/canonical/riddles.jsonl`. The loader
validates each row, rejects duplicate riddle IDs, and sorts rows by community,
topic, optional source row number, and riddle ID.

Prompt context resolves each canonical community through
`data/canonical/corpus_metadata.csv`, using its language, country, and script
family rather than splitting the community key. The root editable installation
resolves this metadata relative to the repository. Custom community keys that
are absent from the metadata retain a generic language/country display fallback;
their script is unspecified. Han metadata maps to `Hani`, because the
script-family field doesn't establish a simplified or traditional variant.

## Run artifacts

Each run creates `data/runs/RUN_ID/` or the directory selected by
`--output-root`. The directory contains:

- `results.jsonl`: one result per processed riddle, including the raw response,
  parsed answer, status, and token counts.
- `run_config.json`: the resolved run configuration.
- `run_metadata.json`: aggregate run counts and timing metadata.
- `manifest.json`: output-file names and checksums.

Framework runs write raw responses. They do not change the frozen score tables
in `data/canonical/`.

## Request serialization

`BenchmarkRunner` passes `reasoning_mode`, `search_enabled`, and
`model_params` to the selected adapter. Adapters control which values reach the
provider. For example, `OpenRouterAdapter` does not serialize
`reasoning_mode`; it serializes `model_params` as request fields. The OpenAI
adapter serializes a truthy `reasoning_mode` as `reasoning_effort`, while the
Anthropic adapter enables adaptive thinking at high effort for a truthy mode.
Check the adapter implementation before interpreting a saved mode as an
effective provider setting.

The canonical model identity and reasoning records are in
`data/canonical/model_mapping.csv` and `data/canonical/model_runs.csv`. The
latter distinguishes historical runner IDs from provider-returned IDs; a
provider-returned ID is unknown when the canonical export did not retain it.
